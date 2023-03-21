from np_pipeline_qc.legacy.copy_d2_lims_files_for_upload import (
    transfer_session,
)

sessions_to_copy = [
    r'\\10.128.54.20\sd8.3\1118324999_576324_20210728',
    r'\\10.128.54.20\sd8.3\1118512505_576324_20210729',
    r'\\10.128.54.20\sd8.3\1119946360_578003_20210804',
    r'\\10.128.54.20\sd8.3\1120251466_578003_20210805',
    r'\\10.128.54.20\sd8.3\1122903357_570302_20210818',
    r'\\10.128.54.20\sd8.3\1123100019_570302_20210819',
]

for s in sessions_to_copy:
    try:
        print('copying {}'.format(s))
        transfer_session(s)
    except Exception as e:
        print('failed to copy {} due to error {}'.format(s, e))
