# change the variables based on the model you are deploying..

preprocessing_script_location = "preprocessing1.py" #no-change
bucket = "fwd-sg-sagemaker-artifacts" #no change
key_prefix="nb_propensity_model" #change to the folder's name

MASTER_FILEPATH = './Prop Data'
TRAIN_FILEPATH = './Data/train'
TEST_FILEPATH = './Data/test'
HYPEROPT_OUTPUT_PATH = './hyperopt_trials'

ID_COL = 'quote_origin_system_number'
LABEL = 'policy_issued_nfc_indicator'
LABELS = [LABEL]

dates_raw = ['quote_issue_date', 'quote_latest_update_date', 'event_created_date', 'applicant_dob', 'first_interaction_date', 'last_interaction_date', 'policy_issue_date']

dates_policy = ['scv_first_policy_issue_date', 'scv_latest_policy_issue_date']

dates_quote = ['first_quote_created_before_date', 'latest_quote_created_before_date']

dates_combined = dates_raw + dates_policy + dates_quote

FEATURES = ['applicant_nric', 'product_type', 'applicant_nric_type', 'applicant_dob', 'applicant_gender', 'applicant_marital_status', 'applicant_occupation', 'applicant_nationality', 'quote_origin_system_number', 'quote_issue_date', 'quote_latest_update_date', 'quote_latest_status', 'quotation_promo_code_value', 'quote_saved_quote_indicator', 'device_type', 'distribution_channel', 'event_id', 'event_created_date', 'quote_valid_flag', 'scv_policy_issued_nfc_count_before', 'scv_policy_cancelled_count_before', 'scv_policy_inforce_count_before', 'scv_policy_gi_count_before', 'scv_policy_li_count_before', 'first_policy_issue_date', 'quote_created_before_count', 'latest_quote_created_before_date', 'first_quote_created_before_date', 'first_interaction_date	last_interaction_date', 'total_interaction_duration', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'nb_payment_failure_call_count', 'nb_payment_failure_no_pick_up_count', 'customer_request_call_count', 'customer_request_no_pick_up_count', 'life_uw_call_count', 'life_uw_no_pick_up_count', 'nb_call_attempt_count', 'assisted_call_flag', 'quote_converted_count', 'quote_converted_indicator', 'policy_issued_nfc_count', 'policy_issued_nfc_indicator', 'policy_issue_date', 'issued_before_interaction_flag']

# FEATURES = ['policy_number', 'policy_status_description', 'product_type', 'policy_plan_code', 'policy_net_premium', 'uw_status', 'uw_processed_flag', 'distribution_channel', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'registered_claim_count', 'issued_nfc_policy_count_before', 'cancelled_policy_count_before', 'avg_promo_code_value_before', 'avg_policy_net_premium_before', 'policy_plan_count_before', 'quote_saved_quote_count_before', 'distribution_channel_count_before', 'device_type_count_before', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'customer_request_call_count', 'customer_request_no_pick_up_count', 'rn_quote_count', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'label', 'renewal_assisted_call_count', 'assisted_call_flag', 'renewal_attempt_call_count', 'no_pickup_call_count', 'policy_plan_category', 'avg_cover_period', 'rn_latest_quote_gap', 'rn_premium_diff', 'previous_premium_diff', 'product_version', 'age_at_quote', 'scv_yor_at_uw']

# FEATURES = ['policy_number', 'policy_status_description', 'product_type', 'policy_plan_code', 'policy_net_premium', 'uw_status', 'uw_processed_flag', 'distribution_channel', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'registered_claim_count', 'issued_nfc_policy_count_before', 'cancelled_policy_count_before', 'avg_promo_code_value_before', 'avg_policy_net_premium_before', 'policy_plan_count_before', 'quote_saved_quote_count_before', 'distribution_channel_count_before', 'device_type_count_before', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'customer_request_call_count', 'customer_request_no_pick_up_count', 'rn_quote_count', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'scv_issued_nfc_policy_count', 'label', 'renewal_assisted_call_count', 'assisted_call_flag', 'renewal_attempt_call_count', 'no_pickup_call_count', 'policy_plan_category', 'avg_cover_period', 'rn_latest_quote_gap', 'rn_premium_diff', 'previous_premium_diff', 'product_version', 'age_at_quote', 'scv_yor_at_uw']

# FEATURES = ['policy_number', 'policy_id', 'policy_status_description', 'policy_issue_date', 'policy_effective_date', 'policy_expiry_date', 'product_type', 'policy_plan_code', 'policy_net_premium', 'uw_date', 'uw_status', 'uw_processed_flag', 'distribution_channel', 'quote_issue_date', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'ph_nric', 'ph_dob', 'registered_claim_count', 'issued_nfc_policy_count_before', 'cancelled_policy_count_before', 'first_policy_issue_date', 'avg_promo_code_value_before', 'avg_policy_net_premium_before', 'policy_plan_count_before', 'quote_saved_quote_count_before', 'distribution_channel_count_before', 'device_type_count_before', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'customer_request_call_count', 'customer_request_no_pick_up_count', 'rn_quote_count', 'rn_latest_quote_issue_date', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'scv_issued_nfc_policy_count', 'scv_first_policy_issue_date', 'rn_policy_id', 'rn_purchase_flow', 'rn_policy_issue_date', 'label', 'policy_expiry_monthyear', 'policy_issue_monthyear', 'first_policy_issue_monthyear', 'policy_expiry_year', 'policy_issue_year', 'renewal_assisted_call_count', 'assisted_call_flag', 'renewal_attempt_call_count', 'no_pickup_call_count', 'policy_plan_category', 'avg_cover_period', 'rn_latest_quote_gap', 'rn_premium_diff', 'previous_premium_diff', 'product_version', 'age_at_quote', 'scv_yor_at_uw']


# FEATURES = ['product_type', 'policy_plan_code', 'policy_gross_premium', 'policy_net_premium', 'uw_status', 'uw_processed_flag', 'distribution_channel', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'registered_claim_count', 'cus_expired_policy_count_before', 'avg_promo_code_value', 'avg_premium_spend', 'first_promo_code_value', 'first_policy_net_premium', 'first_policy_plan_code', 'first_policy_saved_quote_flag', 'first_policy_distribution_channel', 'first_policy_device_type', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'customer_request_call_count', 'customer_request_no_pick_up_count', 'rn_quote_count', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'scv_issued_nfc_policy_count', 'renewal_assisted_call_count', 'renewal_attempt_call_count', 'no_pickup_call_count']

# FEATURES = ['policy_issue_date', 'policy_effective_date', 'policy_expiry_date', 'product_type', 'policy_plan_code', 'policy_gross_premium', 'policy_net_premium', 'uw_date', 'uw_status', 'uw_processed_flag', 'distribution_channel', 'quote_issue_date', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'ph_nric', 'ph_dob', 'registered_claim_count', 'cus_expired_policy_count_before', 'first_policy_issue_date', 'latest_policy_issue_date', 'first_policy_expiry_date', 'latest_policy_expiry_date', 'first_quote_issue_date', 'avg_promo_code_value', 'avg_premium_spend', 'first_promo_code_value', 'first_policy_net_premium', 'first_policy_plan_code', 'first_policy_saved_quote_flag', 'first_policy_distribution_channel', 'first_policy_device_type', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'customer_request_call_count', 'customer_request_no_pick_up_count', 'applicant_nric', 'rn_quote_count', 'rn_latest_quote_issue_date', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'scv_issued_nfc_policy_count', 'scv_first_policy_issue_date', 'rn_policy_id', 'rn_purchase_flow', 'rn_policy_issue_date', 'renewal_assisted_call_count', 'renewal_attempt_call_count', 'no_pickup_call_count', 'policy_expiry_monthyear', 'policy_issue_monthyear', 'first_policy_issue_monthyear', 'policy_expiry_year', 'policy_issue_year', 'first_policy_issue_year']

# FEATURES = ['cover_term', 'product_type', 'policy_plan_code', 'policy_gross_premium', 'policy_net_premium', 'policy_cancellation_flag', 'renewal_rejected_by_customer_flag', 'renewal_invited_clicked_flag', 'distribution_channel', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'insured_age', 'cus_expired_policy_count_before', 'cus_policy_cancellation_count_before', 'cus_renewal_uw_accepted_count_before', 'cus_renewal_rejected_by_customer_count_before', 'cus_renewal_invited_clicked_count_before', 'avg_promo_code_value', 'avg_premium_spend', 'first_promo_code_value', 'first_policy_net_premium', 'first_policy_cover_term', 'first_policy_plan_code', 'first_policy_saved_quote_flag', 'first_policy_distribution_channel', 'first_policy_device_type', 'rn_quote_count', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'scv_issued_nfc_policy_count', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'nb_customer_request_call_count', 'nb_customer_request_no_pick_up_count', 'renewal_customer_request_call_count', 'renewal_customer_request_no_pick_up_count', 'assisted_renewal_call_flag']
# FEATURES = ['cover_term', 'policy_issue_date', 'policy_effective_date', 'policy_expiry_date', 'product_type', 'policy_plan_code', 'policy_gross_premium', 'policy_net_premium', 'uw_date', 'call_planned_date', 'policy_cancellation_flag', 'renewal_rejected_by_customer_flag', 'renewal_invited_clicked_flag', 'distribution_channel', 'quote_issue_date', 'promo_code_value', 'device_type', 'quote_saved_quote_indicator', 'insured_age', 'cus_expired_policy_count_before', 'cus_policy_cancellation_count_before', 'cus_renewal_uw_accepted_count_before', 'cus_renewal_rejected_by_customer_count_before', 'cus_renewal_invited_clicked_count_before', 'first_policy_issue_date', 'latest_policy_issue_date', 'first_policy_expiry_date', 'latest_policy_expiry_date', 'first_quote_issue_date', 'avg_promo_code_value', 'avg_premium_spend', 'first_promo_code_value', 'first_policy_net_premium', 'first_policy_cover_term', 'first_policy_plan_code', 'first_policy_saved_quote_flag', 'first_policy_distribution_channel', 'first_policy_device_type', 'rn_quote_count', 'rn_latest_quote_issue_date', 'rn_saved_quote_count', 'rn_avg_quote_promo_code_value', 'rn_avg_quote_net_premium', 'scv_first_product', 'scv_issued_nfc_policy_count', 'scv_first_policy_issue_date', 'renewal_call_count', 'renewal_no_pick_up_call_count', 'renewal_payment_failure_call_count', 'renewal_payment_failure_no_pick_up_count', 'nb_outbound_call_count', 'nb_outbound_no_pick_up_count', 'nb_customer_request_call_count', 'nb_customer_request_no_pick_up_count', 'renewal_customer_request_call_count', 'renewal_customer_request_no_pick_up_count', 'rn_policy_issue_date', 'rn_policy_status_description', 'rn_flag', 'assisted_renewal_call_flag']

# MODEL_NAME_FREQ_OD = 'base_model_freq_od'
# MODEL_NAME_FREQ_TPPD = 'base_model_freq_tppd_high_trees'
# MODEL_NAME_FREQ_TPBI = 'base_model_freq_tpbi_high_trees'
# MODEL_NAME_SEV_OD = 'base_model_sev_od'
# MODEL_NAME_SEV_TPPD = 'base_model_sev_tppd'
# MODEL_NAME_SEV_TPBI = 'base_model_sev_tpbi'

# ID_COL_FREQ = 'POLICY_NO'
# ID_COL_SEV = 'Claim.No'
# LABEL_SEV_OD = 'Att_Inc_OD'
# LABEL_FREQ_OD = 'Att_cnt_OD'
# LABEL_SEV_TPPD = 'Att_Inc_TPPD'
# LABEL_FREQ_TPPD = 'Att_cnt_TPPD'
# LABEL_SEV_TPBI = 'Att_Inc_TPBI'
# LABEL_FREQ_TPBI = 'Att_cnt_TPBI'
# LABEL_SEV_FT = 'Att_Inc_FT'
# LABEL_FREQ_FT = 'Att_cnt_FT'

# WEIGHT_COL_OD = 'Expo_365_comp'
# WEIGHT_COL_TPPD = 'Expo_365'
# WEIGHT_COL_TPBI = 'Expo_365'

# FEATURES_EXCEL = ['Vehicle Usage', 'Vehicle Make', 'Vehicle Model', 'Gender', 'Drive Exp','CertofMerit', 'NCD', 'Vehicle Age_issued',
#  'AgeBoughtPolicy', 'Claim History 3yr','Vehcc','RiderCnt AuthRider']
# FEATURES_SNAKE = ['vehicle_usage', 'vehicle_make', 'vehicle_model', 'gender', 'drive_exp', 'certofmerit', 'ncd', 'vehicle_age_issued',
#  'ageboughtpolicy', 'claim_history_3yr', 'vehcc', 'ridercnt_authrider']
# # FEATURES is a snake case form of FEATURES_EXCEL
# LABELS = [LABEL_FREQ_OD, LABEL_FREQ_TPPD, LABEL_FREQ_TPBI, LABEL_SEV_OD, LABEL_SEV_TPPD, LABEL_SEV_TPBI ]
# WEIGHT_COLS = [WEIGHT_COL_OD, WEIGHT_COL_TPPD, WEIGHT_COL_TPBI]
# NOT_FEATURES = LABELS + WEIGHT_COLS + [ID_COL_FREQ, ID_COL_SEV]

