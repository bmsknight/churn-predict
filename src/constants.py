DATASET_PATH = "data/train.csv"

DROPPED_COLUMNS = ["state", "area_code"]
NUMERIC_COLUMNS = ["account_length", "number_vmail_messages", "total_day_minutes", "total_day_calls",
                   "total_day_charge", "total_eve_minutes", "total_eve_calls", "total_eve_charge",
                   "total_night_minutes", "total_night_calls", "total_night_charge", "total_intl_minutes",
                   "total_intl_calls", "total_intl_charge", "number_customer_service_calls"]
BINARY_COLUMNS = ["international_plan", "voice_mail_plan", "churn"]
BINARY_MAPPING = {"yes": 1, "no": 0}
TARGET_COLUMN = "churn"
