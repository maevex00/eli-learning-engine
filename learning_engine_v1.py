"""
Eli Learning Engine v2 — Story-driven Campaign Intelligence Dashboard
"""

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
# SL tag specifically).
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

    # Cross-Client Subject Line Analysis patterns (see Subject Line Playbook section).
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
            em.open_rate,
            em.emoji_click_rate,
            em.conversation_rate,
            em.landing_page_opens_text,
            em.landing_page_opens_qr,
            em.video_opens,
            em.unique_opens,
            em.emoji_clicks,
            em.conversation_starts
        FROM subject_line_library sl
        JOIN engagement_metrics em ON em.subject_line_id = sl.id
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

df         = load_data()
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
        "Top Subject Line Emoji Click Rate",
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
    lp   = int(text_data['landing_page_opens_text'].fillna(0).sum())
    vid  = int(text_data['video_opens'].fillna(0).sum())
    conv = int(text_data['conversation_starts'].fillna(0).sum())
    channel_rows.append({
        "Channel":          "Text",
        "Reach":            "—",
        "First Engagement": f"{lp:,} landing page opens",
        "Deep Engagement":  f"{vid:,} video opens" if vid > 0 else "—",
        "Conversion":       f"{conv:,} conversation starts",
        "Campaigns":        len(text_data),
    })

if not qr_data.empty and 'qr' in channels:
    lp_qr = int(qr_data['landing_page_opens_qr'].fillna(0).sum())
    channel_rows.append({
        "Channel":          "QR",
        "Reach":            "—",
        "First Engagement": f"{lp_qr:,} landing page opens",
        "Deep Engagement":  "—",
        "Conversion":       "—",
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
# 4. SUBJECT LINE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

st.header("🧬 Subject Line Intelligence")
st.caption("*What subject line patterns are driving engagement?*")

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

    st.markdown("**A. Basic Subject Line Features**")
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
        st.dataframe(pd.DataFrame(basic_rows)[DISPLAY_COLS], use_container_width=True, hide_index=True)
    else:
        st.info("Not enough campaigns to analyze basic subject line features.")

    st.markdown("**B. Cross-Client Language Patterns**")
    cross_rows = []
    for col, label in CROSS_CLIENT_FEATURES:
        for val, val_label in [(True, 'Yes'), (False, 'No')]:
            sub = feat_df[feat_df[col] == val]
            if len(sub) >= 1:
                cross_rows.append(build_pattern_row(label, val_label, sub))

    if cross_rows:
        st.dataframe(pd.DataFrame(cross_rows)[DISPLAY_COLS], use_container_width=True, hide_index=True)
        if any((r['_emoji_r'] or 0) > 100 for r in cross_rows):
            st.caption(
                "⚠️ Rates above 100% are not mathematically possible — usually a small-sample "
                "pattern (low Campaign Count) hitting unclean/bot-inflated raw counts. Excluded "
                "from the Playbook ranking below."
            )
    else:
        st.info("Not enough campaigns to analyze cross-client language patterns.")

    # ── Subject Line Playbook ─────────────────────────────────────────────────
    st.subheader("📘 Subject Line Playbook")
    st.caption("*Best-performing cross-client patterns, ranked by Emoji Click Rate*")

    INTERPRETATIONS = {
        'Deserve Formula':                    "Positions the voter as someone owed something, not someone being asked.",
        'Listening / Anti-Fundraising Frame': "Reduces skepticism by saying the campaign is here to listen, not ask.",
        'First-Person Frame':                 "Makes the issue feel personal and immediate.",
        'Voter-Focused Frame':                "Centers the voter's life instead of the candidate.",
        'Challenge Frame':                    "Makes the voter the protagonist.",
        'Contrast Frame':                     "Creates a memorable before/after or not-this-but-that structure.",
    }

    # Exclude impossible (>100%) rates the same way Top Performing Subject Lines does —
    # a data-quality artifact, not a real top performer.
    playbook_rows = sorted(
        (r for r in cross_rows if r['Value'] == 'Yes' and r['_emoji_r'] is not None and r['_emoji_r'] <= 100),
        key=lambda r: r['_emoji_r'], reverse=True,
    )

    if playbook_rows:
        playbook_tbl = pd.DataFrame([
            {
                'Pattern':                          r['Pattern'],
                'Campaign Count':                   r['Campaign Count'],
                'Avg Open Rate':                    r['Avg Open Rate'],
                'Avg Emoji Click Rate':             r['Avg Emoji Click Rate'],
                'Avg Conversation Rate':            r['Avg Conversation Rate'],
                'Lift vs Overall Emoji Click Rate': r['Lift vs Overall Emoji Click Rate'],
                'Interpretation':                   INTERPRETATIONS.get(r['Pattern'], "—"),
            }
            for r in playbook_rows
        ])
        st.dataframe(playbook_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data yet to rank subject line patterns.")

    # ── Recommended Next Tests ───────────────────────────────────────────────
    st.subheader("🧪 Recommended Next Tests")

    NEXT_TESTS = {
        'Deserve Formula':                    "Test a more specific voter benefit version, e.g. 'Colorado families deserve safer elections.'",
        'Listening / Anti-Fundraising Frame': "Test 'Not here to fundraise. Here to listen.' for clients where it has not been used.",
        'First-Person Frame':                 "Test first-person economic or safety questions.",
        'Voter-Focused Frame':                "Test direct 'you / your family' phrasing on an axis that currently under-performs.",
        'Challenge Frame':                    "Test a 'prove them wrong' style line for accountability campaigns.",
        'Contrast Frame':                     "Test an explicit 'Not here to X. Here to Y.' structure against the current best subject line for the same axis.",
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

    # ── Axis Performance ──────────────────────────────────────────────────────
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

    # ── Top Performing Subject Lines ──────────────────────────────────────────
    st.subheader("Top Performing Subject Lines")
    st.caption(
        "Eli campaigns only (campaign name starts with \"eli-\" — excludes client-sent campaigns). "
        "Rows with an impossible (>100%) rate are excluded as data-quality artifacts."
    )
    tab1, tab2 = st.tabs(["📬 Open Rate", "😀 Emoji Click Rate"])

    def top_table(pct_col, label, n=10):
        # Rank by the recomputed step-by-step rate (pct_col), not the DB's raw stored
        # rate column — that one is still sent-denominated until the ETL is rerun.
        sub = feat_df[
            feat_df[pct_col].notna() &
            (feat_df[pct_col] <= 100) &
            feat_df['campaign'].str.strip().str.lower().str.startswith(ELI_CAMPAIGN_PREFIX, na=False)
        ].copy()
        top = sub.nlargest(n, pct_col)[
            ['send_date', 'candidate', 'subject_line', 'emails_sent', pct_col]
        ].rename(columns={
            'send_date': 'Date', 'candidate': 'Candidate',
            'subject_line': 'Subject Line', 'emails_sent': 'Sent',
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
        'Date', 'Candidate', 'Channel', 'Subject Line', 'Sent',
        'Open Rate %', 'Emoji Rate %', 'Conv Rate %',
        'LP Opens (Text)', 'LP Opens (QR)', 'Video Opens',
    ]
    available = [(c, n) for c, n in zip(display_cols, col_names) if c in data.columns]
    raw = data[[c for c, _ in available]].copy()
    raw['send_date'] = raw['send_date'].dt.strftime('%Y-%m-%d')
    raw.columns = [n for _, n in available]
    st.dataframe(raw, use_container_width=True, hide_index=True)
