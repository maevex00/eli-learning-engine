"""ETL entry point: James Wiley (Brevo). Wiley moved off MailChimp on 2026-07-14 —
see etl_common.py for the shared data-fetching methodology (Brevo opens/axis + backend
chat_votepayload/chat_userreply for emoji clicks/conversations)."""

import os
import etl_common as common

CANDIDATE        = "James Wiley"
SCHEMA           = "client_253_James_Wiley"
CLIENT_ID        = 253
BREVO_KEY        = os.environ["BREVO_API_KEY"]
CAMPAIGN_PREFIX  = "eli-james_wiley-"

if __name__ == "__main__":
    common.run_brevo(CANDIDATE, SCHEMA, BREVO_KEY, CAMPAIGN_PREFIX, client_id=CLIENT_ID)
