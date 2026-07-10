"""
Eli Learning Engine v2 — Story-driven Campaign Intelligence Dashboard
"""

import html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine

DB_URL = st.secrets["DB_URL"]

# Eli's own campaigns are always named "eli-{candidate}-...(-{Axis}-SL{num}-cv{num})" (see
# extract_axis() in etl_mailchimp_wiley.py). Clients also send their own unrelated campaigns
# (e.g. "KFRW #1", "Sedgwick County ...") on the same dates, so we filter on this naming
# convention instead of a send-date cutoff — a prefix check catches ad-hoc Eli sends that
# never got an SL number too (verified against the DB: 193 "eli-" campaigns vs. 188 with an
# SL tag specifically). Applied dashboard-wide to email-channel rows only, right after
# load_data() — text/qr rows are backend conversations synced by sync_backend_channels() in
# etl_common.py and never carry this naming scheme, so scoping matters (see filter below).
ELI_CAMPAIGN_PREFIX = 'eli-'

st.set_page_config(page_title="Eli Learning Engine", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    [data-testid="stCaptionContainer"] p, .stCaption {
        color: #333333 !important;
        font-size: 1.05rem !important;
    }
    [data-testid="stMetricValue"] { font-size: 2.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 1.05rem !important; color: #222222 !important; }
    [data-testid="stMetricDelta"] { font-size: 1rem !important; }
    h1 { font-size: 2.6rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.5rem !important; }
    [data-testid="stDataFrame"] * { font-size: 1rem !important; }

    table.pattern-table { width: 100%; border-collapse: collapse; margin: 4px 0 14px; font-size: 1rem; }
    table.pattern-table th, table.pattern-table td { padding: 8px 12px; border-bottom: 1px solid #e0e0e0; text-align: left; color: #222222; }
    table.pattern-table thead th { background: #f0f2f6; font-weight: 700; border-bottom: 2px solid #ccc; white-space: nowrap; }
    table.pattern-table tbody tr:hover { background: #f7f9fc; }
    table.pattern-table td.hl-cell { background: #c6efce; color: #046a38; font-weight: 700; }
    table.pattern-table th.q-col, table.pattern-table td.q-col { width: 30px; text-align: center; padding: 8px 4px; }
    table.pattern-table .q-mark {
        display: inline-block; width: 18px; height: 18px; line-height: 18px;
        border-radius: 50%; background: #d8dee8; color: #222222;
        font-size: 0.75rem; font-weight: 700; cursor: help;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

def safe_divide(num, denom):
    if denom and denom > 0 and num is not None and not pd.isna(num):
        return num / denom
    return None

def agg_rate_pct(sub, num_col, denom_col):
    """Sum-of-counts rate (not mean-of-row-rates): rows with a tiny denominator
    would otherwise dominate a simple average and blow the rate past 100%."""
    denom = sub[denom_col].sum()
    if not denom or pd.isna(denom):
        return None
    return sub[num_col].sum() / denom * 100

def format_percent(val, decimals=1):
    if val is None or pd.isna(val):
        return "—"
    return f"{val * 100:.{decimals}f}%"

def extract_subject_line_features(df):
    if df.empty:
        return df
    df = df.copy()
    sl        = df['subject_line'].fillna('')
    candidate = df['candidate'].fillna('') if 'candidate' in df.columns else pd.Series([''] * len(df), index=df.index)

    df['feat_has_question'] = sl.str.contains(r'\?', regex=True)
    df['feat_has_number']   = sl.str.contains(r'\d', regex=True)
    df['feat_has_name'] = [
        any(w in s.lower() for w in c.lower().split() if len(w) > 3)
        for s, c in zip(sl, candidate)
    ]

    df['sl_length'] = sl.str.len()
    df['feat_length_bucket'] = pd.cut(
        df['sl_length'], bins=[0, 40, 70, 999],
        labels=['Short (<40)', 'Medium (40-70)', 'Long (>70)']
    )

    df['feat_word_count'] = sl.str.split().str.len().fillna(0).astype(int)
    df['feat_word_bucket'] = pd.cut(
        df['feat_word_count'], bins=[-1, 8, 12, 999],
        labels=['Short (<=8 words)', 'Medium (9-12 words)', 'Long (>12 words)']
    )

    # Cross-Client Subject Line Analysis patterns (see Headline Playbook section).
    df['feat_contains_deserve'] = sl.str.contains(r'\bdeserves?\b', case=False, regex=True)
    df['feat_listening_frame']  = sl.str.contains(
        r'\blisten\w*\b|\bfundrais\w*\b|\basking\b', case=False, regex=True
    )
    df['feat_first_person']    = sl.str.contains(r'\b(?:i|me|my|we|our)\b', case=False, regex=True)
    df['feat_voter_focused']   = sl.str.contains(
        r'\b(?:you|your|kids|family|families|parents|community)\b', case=False, regex=True
    )
    df['feat_challenge_frame'] = sl.str.contains(
        r'\b(?:prove|wrong|fight|stand up|stop|protect)\b', case=False, regex=True
    )
    df['feat_contrast_frame']  = sl.str.contains(
        r'\b(?:not|but|instead|rather)\b', case=False, regex=True
    )
    return df

def detect_anomalies(df, metric_col, threshold=1.5):
    if df.empty or metric_col not in df.columns:
        return pd.DataFrame()
    series = pd.to_numeric(df[metric_col], errors='coerce').dropna()
    if len(series) < 5:
        return pd.DataFrame()
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.DataFrame()
    df = df.copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
    mask_high = df[metric_col] > mean + threshold * std
    mask_low  = df[metric_col] < mean - threshold * std
    out = df[mask_high | mask_low].copy()
    out['_direction'] = 'Above avg'
    out.loc[mask_low[out.index], '_direction'] = 'Below avg'
    out['_zscore'] = ((out[metric_col] - mean) / std).round(1)
    return out


# ── Pattern-table rendering (Headline Architecture / Keywords / Headline Playbook) ─────────

TOP_N_HIGHLIGHT = 3
RATE_LIFT_NUMERIC_MAP = {
    'Avg Open Rate':                    '_open_r',
    'Avg Emoji Click Rate':             '_emoji_r',
    'Avg Conversation Rate':            '_conv_r',
    'Lift vs Overall Emoji Click Rate': '_lift',
}

def _top_n_indices(rows, numeric_key, n=TOP_N_HIGHLIGHT):
    """Row indices holding the top-n non-null values for numeric_key, for green shading."""
    scored = [(i, r[numeric_key]) for i, r in enumerate(rows) if r.get(numeric_key) is not None]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {i for i, _ in scored[:n]}

def render_pattern_table(rows, display_cols, numeric_map=RATE_LIFT_NUMERIC_MAP):
    """Render a list of row-dicts as an HTML table with:
      - the top 3 values per numeric_map column shaded green
      - a "?" indicator per row whose hover tooltip shows an example headline
    Used for Headline Architecture, Keywords, and the Headline Playbook — the three
    tables the July 8 enhancement brief calls out for this treatment. A plain
    st.dataframe can't do per-cell shading or per-row hover text, hence custom HTML.
    """
    if not rows:
        return
    highlight_sets = {col: _top_n_indices(rows, key) for col, key in numeric_map.items()}

    parts = ['<table class="pattern-table"><thead><tr><th class="q-col"></th>']
    for col in display_cols:
        parts.append(f'<th>{html.escape(col)}</th>')
    parts.append('</tr></thead><tbody>')

    for i, r in enumerate(rows):
        example = r.get('_example_headline')
        tooltip = html.escape(example) if example else 'No example headline available'
        parts.append('<tr>')
        parts.append(f'<td class="q-col"><span class="q-mark" title="{tooltip}">?</span></td>')
        for col in display_cols:
            cls = ' class="hl-cell"' if i in highlight_sets.get(col, set()) else ''
            parts.append(f'<td{cls}>{html.escape(str(r[col]))}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    engine = create_engine(DB_URL)
    df = pd.read_sql("""
        SELECT
            sl.id,
            sl.candidate,
            sl.campaign,
            sl.subject_line,
            sl.channel,
            sl.send_date,
            sl.emails_sent,
            sl.axis_primary,
            sl.cta_text,
            em.open_rate,
            em.emoji_click_rate,
            em.conversation_rate,
            em.landing_page_opens_text,
            em.landing_page_opens_qr,
            em.video_opens,
            em.unique_opens,
            em.emoji_clicks,
            em.conversation_starts,
            ec.landing_page_greeting_video_url AS video_url
        FROM subject_line_library sl
        JOIN engagement_metrics em ON em.subject_line_id = sl.id
        LEFT JOIN eli_conversation ec ON ec.id = sl.conversation_id
        ORDER BY sl.candidate, sl.send_date
    """, engine)
    df['send_date'] = pd.to_datetime(df['send_date'])

    # Recompute step-by-step funnel rates from raw counts rather than trusting the stored
    # rate columns: those were historically computed against total emails_sent at every
    # stage (e.g. emoji_click_rate = clicks / sent) instead of against the previous stage.
    df['open_rate_pct']         = (df['unique_opens']       / df['emails_sent'].replace(0, np.nan) * 100).round(2)
    df['emoji_click_rate_pct']  = (df['emoji_clicks']        / df['unique_opens'].replace(0, np.nan) * 100).round(2)
    df['conversation_rate_pct'] = (df['conversation_starts'] / df['emoji_clicks'].replace(0, np.nan) * 100).round(2)
    return df

# Engagements that have ended — kept in the database for future cross-client
# learning, but no longer shown on this dashboard.
RETIRED_CANDIDATES = {"Joy Eakins", "John Czajka"}

df = load_data()
df = df[~df['candidate'].isin(RETIRED_CANDIDATES)]

# Data scope: only Eli Works-sent campaigns (see ELI_CAMPAIGN_PREFIX above), applied
# dashboard-wide. Scoped to channel == 'email' — text/qr rows are backend conversations,
# not MailChimp campaigns, and their `campaign` value (a conversation label) never carries
# the "eli-...-SL..." naming scheme, so an unscoped filter would wipe out all text/qr data.
df = df[
    (df['channel'] != 'email') |
    df['campaign'].fillna('').str.strip().str.lower().str.startswith(ELI_CAMPAIGN_PREFIX)
]

candidates = sorted(df['candidate'].unique().tolist())


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🧠 Eli Learning Engine")
st.sidebar.markdown("---")

view = st.sidebar.radio("View", ["Overall", "Client View"])
if view == "Client View":
    selected_client = st.sidebar.selectbox("Candidate", candidates)
    data = df[df['candidate'] == selected_client].copy()
else:
    selected_client = None
    data = df.copy()

st.sidebar.markdown("---")
all_channels = sorted(df['channel'].dropna().unique().tolist())
channels = st.sidebar.multiselect("Channel", all_channels, default=all_channels)
data = data[data['channel'].isin(channels)]

email_data = data[data['channel'] == 'email'].copy()
text_data  = data[data['channel'] == 'text'].copy()
qr_data    = data[data['channel'] == 'qr'].copy()


# ── Header ────────────────────────────────────────────────────────────────────

title = f"🧠 {selected_client} — Learning Engine" if view == "Client View" else "🧠 Eli Learning Engine"
st.title(title)
st.caption(
    f"{len(data)} campaigns · {data['candidate'].nunique()} candidate(s) · "
    "MailChimp + Eli Intel"
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📊 Snapshot")
st.caption("*We're not in the email business — we're in the business of clicks and conversations.*")

if not email_data.empty and 'email' in channels:
    email_all = df[df['channel'] == 'email']
    overall_avg_emoji = agg_rate_pct(email_all, 'emoji_clicks', 'unique_opens')
    overall_avg_conv  = agg_rate_pct(email_all, 'conversation_starts', 'emoji_clicks')

    avg_emoji = agg_rate_pct(email_data, 'emoji_clicks', 'unique_opens')
    avg_conv  = agg_rate_pct(email_data, 'conversation_starts', 'emoji_clicks')
    n_camps   = len(email_data)

    ch_agg = data.groupby('channel').apply(
        lambda g: agg_rate_pct(g, 'emoji_clicks', 'unique_opens'), include_groups=False
    ).dropna()
    best_channel = ch_agg.idxmax().title() if not ch_agg.empty else "—"

    # Exclude rows with an impossible (>100%) rate — a data-quality artifact from
    # unclean/bot-inflated raw counts (see Known Issues), not a real top performer.
    sane_rows     = email_data[email_data['emoji_click_rate_pct'] <= 100]
    best_row      = sane_rows.nlargest(1, 'emoji_click_rate_pct')
    best_sl_label = best_row['subject_line'].iloc[0][:45] + "…" if not best_row.empty else "—"
    best_sl_rate  = best_row['emoji_click_rate_pct'].iloc[0] if not best_row.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Campaigns Analyzed", n_camps)

    avg_emoji_str = f"{avg_emoji:.2f}%" if avg_emoji is not None else "—"
    avg_conv_str  = f"{avg_conv:.2f}%"  if avg_conv  is not None else "—"

    if view == "Client View" and avg_emoji is not None and overall_avg_emoji is not None:
        delta_emoji = round(avg_emoji - overall_avg_emoji, 2)
        delta_conv  = round(avg_conv - overall_avg_conv, 2) if avg_conv is not None and overall_avg_conv is not None else None
        c2.metric("Avg Emoji Click Rate",  avg_emoji_str, delta=f"{delta_emoji:+.2f}% vs overall")
        c3.metric("Avg Conversation Rate", avg_conv_str,  delta=f"{delta_conv:+.2f}% vs overall" if delta_conv is not None else None)
    else:
        c2.metric("Avg Emoji Click Rate",  avg_emoji_str)
        c3.metric("Avg Conversation Rate", avg_conv_str)

    c4.metric("Best Channel", best_channel)
    c5.metric(
        "Top Headline Emoji Click Rate",
        f"{best_sl_rate:.2f}%" if best_sl_rate else "—",
        help=best_sl_label,
    )

else:
    st.info("No email data available for the current selection.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENGAGEMENT FUNNEL
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🔽 Engagement Funnel")
st.caption("*Where are people dropping off?*")

if not email_data.empty and 'email' in channels:
    total_sent  = int(email_data['emails_sent'].fillna(0).sum())
    total_opens = int(email_data['unique_opens'].fillna(0).sum())
    total_emoji = int(email_data['emoji_clicks'].fillna(0).sum())
    total_conv  = int(email_data['conversation_starts'].fillna(0).sum())

    open_rate_  = safe_divide(total_opens, total_sent)
    emoji_rate_ = safe_divide(total_emoji, total_opens)

    funnel_labels = ["Emails Sent", "Unique Opens", "Emoji Clicks", "Conversation Starts"]
    funnel_values = [total_sent, total_opens, total_emoji, total_conv]

    conv_rate_ = safe_divide(total_conv, total_emoji)

    # Custom text: first step shows absolute count only; subsequent steps show step-to-step rate
    custom_text = [
        f"<b>{total_sent:,}</b>",
        f"<b>{total_opens:,}</b><br>{format_percent(open_rate_)} of sent",
        f"<b>{total_emoji:,}</b><br>{format_percent(emoji_rate_)} of openers",
        f"<b>{total_conv:,}</b><br>{format_percent(conv_rate_)} of clickers",
    ]

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_labels,
        x=funnel_values,
        text=custom_text,
        textinfo="text",
        textfont=dict(size=24, color="white", family="Arial Black"),
        marker=dict(color=["#1565C0", "#2E7D32", "#E65100", "#B71C1C"]),
        connector=dict(line=dict(color="rgba(0,0,0,0.25)", width=2)),
    ))
    fig_funnel.update_layout(
        height=750,
        margin=dict(t=40, b=40, l=260, r=60),
        font=dict(size=20, color="#111111"),
        paper_bgcolor="white",
        yaxis=dict(tickfont=dict(size=22, color="#111111")),
    )
    st.plotly_chart(fig_funnel, use_container_width=True, config={"scrollZoom": True})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📤 Emails Sent",        f"{total_sent:,}")
    col2.metric("👁 Unique Opens",        f"{total_opens:,}",
                delta=f"{format_percent(open_rate_)} open rate", delta_color="off")
    col3.metric("😊 Emoji Clicks",        f"{total_emoji:,}",
                delta=f"{format_percent(emoji_rate_)} of openers", delta_color="off")
    col4.metric("💬 Conversation Starts", f"{total_conv:,}",
                delta=f"{format_percent(conv_rate_)} of emoji clickers", delta_color="off")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHANNEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

st.header("📡 Channel Comparison")
st.caption("*Which channel drives the most engagement?*")

channel_rows = []

if not email_data.empty and 'email' in channels:
    email_open_rate  = agg_rate_pct(email_data, 'unique_opens', 'emails_sent')
    email_emoji_rate = agg_rate_pct(email_data, 'emoji_clicks', 'unique_opens')
    channel_rows.append({
        "Channel":          "Email",
        "Reach":            f"{int(email_data['emails_sent'].fillna(0).sum()):,} sent",
        "First Engagement": (f"{email_open_rate:.1f}% open rate  |  " if email_open_rate is not None else "") +
                            f"{int(email_data['unique_opens'].fillna(0).sum()):,} unique opens",
        "Deep Engagement":  (f"{email_emoji_rate:.2f}% emoji click rate  |  " if email_emoji_rate is not None else "") +
                            f"{int(email_data['emoji_clicks'].fillna(0).sum()):,} clicks",
        "Conversion":       f"{int(email_data['conversation_starts'].fillna(0).sum()):,} conversation starts",
        "Campaigns":        len(email_data),
    })

if not text_data.empty and 'text' in channels:
    lp    = int(text_data['landing_page_opens_text'].fillna(0).sum())
    emoji = int(text_data['emoji_clicks'].fillna(0).sum())
    conv  = int(text_data['conversation_starts'].fillna(0).sum())
    channel_rows.append({
        "Channel":          "Text",
        "Reach":            "—",
        "First Engagement": f"{lp:,} landing page opens",
        "Deep Engagement":  f"{emoji:,} emoji clicks" if emoji else "—",
        "Conversion":       f"{conv:,} conversation starts",
        "Campaigns":        len(text_data),
    })

if not qr_data.empty and 'qr' in channels:
    lp_qr = int(qr_data['landing_page_opens_qr'].fillna(0).sum())
    emoji = int(qr_data['emoji_clicks'].fillna(0).sum())
    conv  = int(qr_data['conversation_starts'].fillna(0).sum())
    channel_rows.append({
        "Channel":          "QR",
        "Reach":            "—",
        "First Engagement": f"{lp_qr:,} landing page opens",
        "Deep Engagement":  f"{emoji:,} emoji clicks" if emoji else "—",
        "Conversion":       f"{conv:,} conversation starts" if conv else "—",
        "Campaigns":        len(qr_data),
    })

# Generic fallback for channels beyond email/text/qr (e.g. Website, sourced from
# vote_payload.channel) so new channels show up without code changes.
KNOWN_CHANNELS = {"email", "text", "qr"}
for ch in sorted(set(data['channel'].dropna().unique()) - KNOWN_CHANNELS):
    sub = data[data['channel'] == ch]
    if sub.empty:
        continue
    opens = int(sub['unique_opens'].fillna(0).sum()) if 'unique_opens' in sub else 0
    emoji = int(sub['emoji_clicks'].fillna(0).sum()) if 'emoji_clicks' in sub else 0
    conv  = int(sub['conversation_starts'].fillna(0).sum()) if 'conversation_starts' in sub else 0
    channel_rows.append({
        "Channel":          ch.title(),
        "Reach":            "—",
        "First Engagement": f"{opens:,} opens" if opens else "—",
        "Deep Engagement":  f"{emoji:,} emoji clicks" if emoji else "—",
        "Conversion":       f"{conv:,} conversation starts" if conv else "—",
        "Campaigns":        len(sub),
    })

if channel_rows:
    st.dataframe(pd.DataFrame(channel_rows), use_container_width=True, hide_index=True)
else:
    st.info("No data for the selected channels.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HEADLINE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🧬 Headline Intelligence")
st.caption("*What headline patterns are driving engagement?*")

if not email_data.empty and 'email' in channels:
    feat_df = extract_subject_line_features(email_data)

    # ── Pattern Analysis ──────────────────────────────────────────────────────
    st.subheader("Pattern Analysis")

    # Reference point for "lift": the overall emoji click rate across every campaign
    # currently in view (respects the sidebar's client/channel filters).
    overall_emoji_rate = agg_rate_pct(feat_df, 'emoji_clicks', 'unique_opens')

    def build_pattern_row(pattern_label, value_label, sub):
        open_r  = agg_rate_pct(sub, 'unique_opens', 'emails_sent')
        emoji_r = agg_rate_pct(sub, 'emoji_clicks', 'unique_opens')
        conv_r  = agg_rate_pct(sub, 'conversation_starts', 'emoji_clicks')
        lift    = (emoji_r - overall_emoji_rate) if (emoji_r is not None and overall_emoji_rate is not None) else None

        # Example headline for this row's "?" tooltip: the best-performing (and sane,
        # i.e. <=100%) headline among the campaigns that make up this pattern/value.
        sane_sub      = sub[sub['emoji_click_rate_pct'].notna() & (sub['emoji_click_rate_pct'] <= 100)]
        example_row   = sane_sub.nlargest(1, 'emoji_click_rate_pct')
        example_headline = example_row['subject_line'].iloc[0] if not example_row.empty else None

        return {
            'Pattern':                          pattern_label,
            'Value':                            value_label,
            'Campaign Count':                   len(sub),
            'Avg Open Rate':                    f"{open_r:.1f}%"  if open_r  is not None else "—",
            'Avg Emoji Click Rate':             f"{emoji_r:.2f}%" if emoji_r is not None else "—",
            'Avg Conversation Rate':            f"{conv_r:.2f}%"  if conv_r  is not None else "—",
            'Lift vs Overall Emoji Click Rate': f"{lift:+.2f} pp" if lift    is not None else "—",
            # Raw numerics kept for sorting/playbook/next-test logic below; not displayed.
            '_open_r': open_r, '_emoji_r': emoji_r, '_conv_r': conv_r, '_lift': lift,
            '_example_headline': example_headline,
        }

    BASIC_BOOL_FEATURES = [
        ('feat_has_question', 'Contains Question'),
        ('feat_has_number',   'Contains Number'),
        ('feat_has_name',     'Contains Candidate Name'),
    ]
    CROSS_CLIENT_FEATURES = [
        ('feat_contains_deserve', 'Deserve Formula'),
        ('feat_listening_frame',  'Listening / Anti-Fundraising Frame'),
        ('feat_first_person',     'First-Person Frame'),
        ('feat_voter_focused',    'Voter-Focused Frame'),
        ('feat_challenge_frame',  'Challenge Frame'),
        ('feat_contrast_frame',   'Contrast Frame'),
    ]
    DISPLAY_COLS = [
        'Pattern', 'Value', 'Campaign Count', 'Avg Open Rate',
        'Avg Emoji Click Rate', 'Avg Conversation Rate', 'Lift vs Overall Emoji Click Rate',
    ]

    # Plain-English definitions for the Cross-Client patterns — reused by the Keywords
    # definitions table, the Headline Playbook's Interpretation column, and Recommended
    # Next Tests' "Why it matters" column.
    INTERPRETATIONS = {
        'Deserve Formula':                    "Positions the voter as someone owed something, not someone being asked.",
        'Listening / Anti-Fundraising Frame': "Reduces skepticism by saying the campaign is here to listen, not ask.",
        'First-Person Frame':                 "Makes the issue feel personal and immediate.",
        'Voter-Focused Frame':                "Centers the voter's life instead of the candidate.",
        'Challenge Frame':                    "Makes the voter the protagonist.",
        'Contrast Frame':                     "Creates a memorable before/after or not-this-but-that structure.",
    }

    st.markdown("**Headline Architecture**")
    basic_rows = []
    for col, label in BASIC_BOOL_FEATURES:
        for val, val_label in [(True, 'Yes'), (False, 'No')]:
            sub = feat_df[feat_df[col] == val]
            if len(sub) >= 2:
                basic_rows.append(build_pattern_row(label, val_label, sub))

    for bucket in ['Short (<40)', 'Medium (40-70)', 'Long (>70)']:
        sub = feat_df[feat_df['feat_length_bucket'] == bucket]
        if len(sub) >= 2:
            basic_rows.append(build_pattern_row('Character Length Bucket', bucket, sub))

    for bucket in ['Short (<=8 words)', 'Medium (9-12 words)', 'Long (>12 words)']:
        sub = feat_df[feat_df['feat_word_bucket'] == bucket]
        if len(sub) >= 2:
            basic_rows.append(build_pattern_row('Word Count Bucket', bucket, sub))

    if basic_rows:
        render_pattern_table(basic_rows, DISPLAY_COLS)
    else:
        st.info("Not enough campaigns to analyze headline architecture.")

    st.markdown("**Keywords**")
    st.caption("What each pattern below means, in plain English, before you hit the numbers:")
    st.table(pd.DataFrame(
        [{"Term": k, "Definition": v} for k, v in INTERPRETATIONS.items()]
    ))

    cross_rows = []
    for col, label in CROSS_CLIENT_FEATURES:
        for val, val_label in [(True, 'Yes'), (False, 'No')]:
            sub = feat_df[feat_df[col] == val]
            if len(sub) >= 1:
                cross_rows.append(build_pattern_row(label, val_label, sub))

    if cross_rows:
        render_pattern_table(cross_rows, DISPLAY_COLS)
        if any((r['_emoji_r'] or 0) > 100 for r in cross_rows):
            st.caption(
                "⚠️ Rates above 100% are not mathematically possible — usually a small-sample "
                "pattern (low Campaign Count) hitting unclean/bot-inflated raw counts. Excluded "
                "from the Playbook ranking below."
            )
    else:
        st.info("Not enough campaigns to analyze keywords.")

    # ── Headline Playbook ─────────────────────────────────────────────────────
    st.subheader("📘 Headline Playbook")
    st.caption("*Best-performing cross-client patterns, ranked by Emoji Click Rate*")

    # Exclude impossible (>100%) rates the same way Top Headlines does —
    # a data-quality artifact, not a real top performer.
    playbook_rows = sorted(
        (r for r in cross_rows if r['Value'] == 'Yes' and r['_emoji_r'] is not None and r['_emoji_r'] <= 100),
        key=lambda r: r['_emoji_r'], reverse=True,
    )

    if playbook_rows:
        playbook_render_rows = []
        for r in playbook_rows:
            rr = dict(r)
            rr['Interpretation'] = INTERPRETATIONS.get(r['Pattern'], "—")
            playbook_render_rows.append(rr)
        render_pattern_table(
            playbook_render_rows,
            ['Pattern', 'Campaign Count', 'Avg Open Rate', 'Avg Emoji Click Rate',
             'Avg Conversation Rate', 'Lift vs Overall Emoji Click Rate', 'Interpretation'],
        )
    else:
        st.info("Not enough data yet to rank headline patterns.")

    # ── Table C: CTA Analysis ──────────────────────────────────────────────────
    st.subheader("Table C: CTA Analysis")
    st.caption("*Which calls-to-action are driving engagement?*")

    cta_df = feat_df[feat_df['cta_text'].notna() & (feat_df['cta_text'].str.strip() != '')]
    if not cta_df.empty:
        cta_rows = []
        for cta_text, sub in cta_df.groupby('cta_text'):
            open_r  = agg_rate_pct(sub, 'unique_opens', 'emails_sent')
            emoji_r = agg_rate_pct(sub, 'emoji_clicks', 'unique_opens')
            conv_r  = agg_rate_pct(sub, 'conversation_starts', 'emoji_clicks')
            lift    = (emoji_r - overall_emoji_rate) if (emoji_r is not None and overall_emoji_rate is not None) else None
            cta_rows.append({
                'CTA':                               cta_text if len(cta_text) <= 90 else cta_text[:87] + '…',
                'Campaign Count':                    len(sub),
                'Avg Open Rate':                      f"{open_r:.1f}%"  if open_r  is not None else "—",
                'Avg Emoji Click Rate':                f"{emoji_r:.2f}%" if emoji_r is not None else "—",
                'Avg Conversation Rate':               f"{conv_r:.2f}%"  if conv_r  is not None else "—",
                'Lift vs Overall Emoji Click Rate':    f"{lift:+.2f} pp" if lift    is not None else "—",
                '_sort_emoji_r':                       emoji_r if emoji_r is not None else -1,
            })
        cta_rows.sort(key=lambda r: r['_sort_emoji_r'], reverse=True)
        cta_tbl = pd.DataFrame(cta_rows).drop(columns=['_sort_emoji_r'])
        st.dataframe(cta_tbl, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No CTA text extracted yet for the current selection — CTA text is pulled from "
            "each campaign's MailChimp HTML (EMOTE PROMPT/CTA section) during the nightly ETL; "
            "it backfills automatically as campaigns get (re-)processed."
        )

    st.markdown("**By Landing Page Video Presence**")
    if not cta_df.empty:
        has_video = cta_df['video_url'].notna() & (cta_df['video_url'].str.strip() != '')
        st.table(pd.DataFrame([
            {"Video on Landing Page": "Video Present",     "Campaign Count": int(has_video.sum())},
            {"Video on Landing Page": "Video Not Present", "Campaign Count": int((~has_video).sum())},
        ]))
    else:
        st.info("No CTA campaigns in the current selection to break down by video presence.")

    # ── Axis Performance ──────────────────────────────────────────────────────
    # Positioned here — right after Table C — per the July 8 enhancement brief.
    if feat_df['axis_primary'].notna().any():
        st.subheader("Axis Performance — Which Axis Converts Opens → Clicks → Conversations?")
        st.caption("*Step-by-step funnel conversion rates computed from raw counts per Axis*")

        axis_src = feat_df[feat_df['axis_primary'].notna()].copy()
        axis_raw = (
            axis_src.groupby('axis_primary')
            .agg(
                campaigns=('id', 'count'),
                emails_sent=('emails_sent', 'sum'),
                unique_opens=('unique_opens', 'sum'),
                emoji_clicks=('emoji_clicks', 'sum'),
                conversation_starts=('conversation_starts', 'sum'),
            )
            .reset_index()
        )
        axis_raw['Open Rate\n(Sent→Open)']    = (axis_raw['unique_opens']        / axis_raw['emails_sent'].replace(0, np.nan) * 100).round(1)
        axis_raw['Click Rate\n(Open→Emoji)']  = (axis_raw['emoji_clicks']         / axis_raw['unique_opens'].replace(0, np.nan) * 100).round(1)
        axis_raw['Conv Rate\n(Emoji→Conv)']   = (axis_raw['conversation_starts']  / axis_raw['emoji_clicks'].replace(0, np.nan) * 100).round(1)
        # Sort on the final funnel stage — conversion is the story, not open rate.
        axis_raw = axis_raw.sort_values('Conv Rate\n(Emoji→Conv)', ascending=False)

        axis_melt = axis_raw.melt(
            id_vars='axis_primary',
            value_vars=['Open Rate\n(Sent→Open)', 'Click Rate\n(Open→Emoji)', 'Conv Rate\n(Emoji→Conv)'],
            var_name='Metric', value_name='Rate (%)',
        )
        fig_axis = px.bar(
            axis_melt,
            x='axis_primary',
            y='Rate (%)',
            color='Metric',
            barmode='group',
            labels={'axis_primary': 'Axis'},
            color_discrete_map={
                'Open Rate\n(Sent→Open)':   '#1565C0',
                'Click Rate\n(Open→Emoji)': '#E65100',
                'Conv Rate\n(Emoji→Conv)':  '#2E7D32',
            },
            text='Rate (%)',
        )
        fig_axis.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_size=16)
        fig_axis.update_layout(
            height=560,
            margin=dict(t=40, b=20),
            font=dict(size=18, color="#111111"),
            legend=dict(font=dict(size=16), title_font_size=16),
            yaxis=dict(title="Rate (%)", tickfont=dict(size=16)),
            xaxis=dict(tickfont=dict(size=17)),
        )
        st.plotly_chart(fig_axis, use_container_width=True, config={"scrollZoom": True})

        # Full funnel table per Axis
        tbl = axis_raw[[
            'axis_primary', 'campaigns', 'emails_sent', 'unique_opens',
            'emoji_clicks', 'conversation_starts',
            'Open Rate\n(Sent→Open)', 'Click Rate\n(Open→Emoji)', 'Conv Rate\n(Emoji→Conv)',
        ]].copy()
        tbl.columns = [
            'Axis', 'Campaigns', 'Sent', 'Opens', 'Emoji Clicks', 'Conversations',
            'Open Rate %', 'Open→Emoji %', 'Emoji→Conv %',
        ]
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        if (axis_raw[['Open Rate\n(Sent→Open)', 'Click Rate\n(Open→Emoji)', 'Conv Rate\n(Emoji→Conv)']] > 100).any().any():
            st.caption(
                "⚠️ Rates above 100% are not mathematically possible — the raw counts feeding "
                "them include unclean/bot traffic upstream. Treat as directional only until the "
                "Backend bot-cleaner output is what this dashboard reads from."
            )

    # ── Top Headlines ──────────────────────────────────────────────────────────
    st.subheader("Top Headlines")
    st.caption("Rows with an impossible (>100%) rate are excluded as data-quality artifacts.")
    tab1, tab2, tab3 = st.tabs(["📬 Open Rate", "😀 Emoji Click Rate", "💬 Conversation Rate"])

    def top_table(pct_col, label, n=10):
        # Rank by the recomputed step-by-step rate (pct_col), not the DB's raw stored
        # rate column — that one is still sent-denominated until the ETL is rerun.
        # (Eli-campaign scoping now happens once, globally, right after load_data().)
        sub = feat_df[
            feat_df[pct_col].notna() &
            (feat_df[pct_col] <= 100)
        ].copy()
        top = sub.nlargest(n, pct_col)[
            ['send_date', 'candidate', 'subject_line', 'emails_sent', pct_col]
        ].rename(columns={
            'send_date': 'Date', 'candidate': 'Candidate',
            'subject_line': 'Headline', 'emails_sent': 'Sent',
            pct_col: label,
        })
        top['Date'] = top['Date'].dt.strftime('%Y-%m-%d')
        top[label]  = top[label].apply(lambda x: f"{x:.2f}%")
        return top.reset_index(drop=True)

    with tab1:
        st.dataframe(top_table('open_rate_pct', 'Open Rate'),
                     use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(top_table('emoji_click_rate_pct', 'Emoji Click Rate'),
                     use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(top_table('conversation_rate_pct', 'Conversation Rate'),
                     use_container_width=True, hide_index=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📋 Raw Data"):
    display_cols = [
        'send_date', 'candidate', 'channel', 'subject_line', 'emails_sent',
        'open_rate_pct', 'emoji_click_rate_pct', 'conversation_rate_pct',
        'landing_page_opens_text', 'landing_page_opens_qr', 'video_opens',
    ]
    col_names = [
        'Date', 'Candidate', 'Channel', 'Headline', 'Sent',
        'Open Rate %', 'Emoji Rate %', 'Conv Rate %',
        'LP Opens (Text)', 'LP Opens (QR)', 'Video Opens',
    ]
    available = [(c, n) for c, n in zip(display_cols, col_names) if c in data.columns]
    raw = data[[c for c, _ in available]].copy()
    raw['send_date'] = raw['send_date'].dt.strftime('%Y-%m-%d')
    raw.columns = [n for _, n in available]
    st.dataframe(raw, use_container_width=True, hide_index=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RECOMMENDED NEXT TESTS  (kept last — nothing appears after this section)
# ═══════════════════════════════════════════════════════════════════════════════

if not email_data.empty and 'email' in channels:
    st.header("🧪 Recommended Next Tests")

    NEXT_TESTS = {
        'Deserve Formula':                    "Test a more specific voter benefit version, e.g. 'Colorado families deserve safer elections.'",
        'Listening / Anti-Fundraising Frame': "Test 'Not here to fundraise. Here to listen.' for clients where it has not been used.",
        'First-Person Frame':                 "Test first-person economic or safety questions.",
        'Voter-Focused Frame':                "Test direct 'you / your family' phrasing on an axis that currently under-performs.",
        'Challenge Frame':                    "Test a 'prove them wrong' style line for accountability campaigns.",
        'Contrast Frame':                     "Test an explicit 'Not here to X. Here to Y.' structure against the current best headline for the same axis.",
    }

    next_test_rows = []
    for r in cross_rows:
        if r['Value'] != 'Yes':
            continue
        n, emoji_r, lift = r['Campaign Count'], r['_emoji_r'], r['_lift']
        if emoji_r is not None and emoji_r > 100:
            status = "Data quality issue — exclude until raw counts are clean"
        elif n < 5 and emoji_r is not None and overall_emoji_rate is not None and emoji_r > overall_emoji_rate:
            status = "Promising — needs more testing"
        elif n >= 5 and lift is not None and lift > 0:
            status = "Validated pattern"
        elif lift is not None and lift < 0:
            status = "Avoid or rewrite"
        else:
            status = "Inconclusive — insufficient signal"
        next_test_rows.append({
            'Pattern':             r['Pattern'],
            'Status':              status,
            'Why it matters':      INTERPRETATIONS.get(r['Pattern'], "—"),
            'Suggested next test': NEXT_TESTS.get(r['Pattern'], "—"),
        })

    if next_test_rows:
        st.dataframe(pd.DataFrame(next_test_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data yet to recommend next tests.")
