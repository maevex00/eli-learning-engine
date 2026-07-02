"""ETL entry point: Joy Eakins. See etl_common.py for the shared data-fetching
methodology (MailChimp opens/axis + backend chat_votepayload/chat_userreply for
emoji clicks/conversations, matching eliworks-mailchimp-dashboard/app.py)."""

import os
import etl_common as common

CANDIDATE = "Joy Eakins"
SCHEMA    = "client_49_Joy_Eakins"
CLIENT_ID = 49
MC_KEY    = os.environ["JOY_MC_KEY"]
MC_DC     = "us7"

if __name__ == "__main__":
    common.run(CANDIDATE, SCHEMA, MC_KEY, MC_DC, client_id=CLIENT_ID)
