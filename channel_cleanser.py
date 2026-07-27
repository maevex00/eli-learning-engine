"""
Channel-aware emoji click aggregation.

eli_intel.emoji_clicks / eli_intelemoji (rebuilt by the production Vote Cleanser's
rebuild_intel()) are conversation-level totals across ALL channels combined — that
function's GROUP BY has no channel column. A conversation_id shared across channels
(e.g. email + event + unknown) gets its clicks summed together there, which is what
caused conv 969 (2026-07-14): 18 combined clicks (7 email + 1 event + 10 unknown)
were all credited to the email row despite it having 0 opens.

This module does NOT re-run bot detection — the Vote Cleanser's vote_status column
(written every 15 minutes onto {schema}.chat_votepayload) is still the single source
of truth for what counts as a bot. This only re-aggregates its already-tagged 'valid'
rows, with `channel` added to the GROUP BY, into eli_intel_channel — the table
etl_common.py reads emoji clicks from instead of chat_votepayload directly.
"""

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eli_intel_channel (
    id              SERIAL PRIMARY KEY,
    client_id       INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL,
    channel         TEXT NOT NULL,
    emoji_clicks    INTEGER NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (client_id, conversation_id, channel)
)
"""


def ensure_table(conn):
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.close()


def rebuild_channel_emoji_clicks(conn, schema, client_id):
    """Full rebuild of this client's per-(conversation, channel) valid emoji click
    counts, from {schema}.chat_votepayload.vote_status (written by the production
    Vote Cleanser — never re-derived here, only re-grouped by channel).

    Delete+reinsert per client (same pattern as write_updates()'s conversation_starts
    clear in etl_common.py) so a (conversation, channel) pair that dropped to zero
    since the last run — e.g. everything on that channel got purged — doesn't leave
    a stale count behind.
    """
    ensure_table(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT conversation_id, channel, COUNT(*)
        FROM "{schema}".chat_votepayload
        WHERE vote_status = 'valid'
        GROUP BY conversation_id, channel
    """.format(schema=schema))
    rows = cur.fetchall()

    cur.execute("DELETE FROM eli_intel_channel WHERE client_id = %s", (client_id,))
    if rows:
        cur.executemany("""
            INSERT INTO eli_intel_channel (client_id, conversation_id, channel, emoji_clicks, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, [(client_id, conv_id, channel, cnt) for conv_id, channel, cnt in rows])

    conn.commit()
    cur.close()
    print(f"  eli_intel_channel rebuilt: {len(rows)} (conversation, channel) row(s) for client {client_id}")
    return len(rows)
