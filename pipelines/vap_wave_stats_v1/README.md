# VAP_Wave_Stats_v1 Transformation Pipeline

This VAP pipeline works in conjuction with the `vap_wave_raw_v1` and `vap_gps_v1` VAP pipelines to 
combine and plot the buoy's wave and position data to create the final data product.

Items to verify:
 - Make sure the correct spotter dataastream is set in `vap_wave_stats_v1/config/pipeline.yaml`
 - Make sure the "begin" and "end" timestamps are sequential and exist in the `a1` datastream being run
 - Make sure to adjust the constants dictionary in `shared/wave_analysis.py`
 - Make sure QC values in `vap_wave_stats_v1/config/dataset.yaml` are in the expected range
 - Make sure the `time_padding` parameter in retriever.yaml is enough to cover the time difference between your start time and the time in the `a1` filename


## Prerequisites

* Ensure that your development environment has been set up according to
[the instructions](../../README.md#development-environment-setup).

## Running your pipeline

1. Navigate to the repository root from the terminal (i.e., 2 levels up from this file)
2. Run `runner.py` and specify the transformation pipeline that should run:

        ```shell
        python runner.py vap pipelines/vap_wave_stats_v1/config/pipeline.yaml -b 20230324 -e 20230325
        ```


## Testing your pipeline
This template is set up with a pytest unit test to ensure your pipeline is working correctly.  It is intended that the
pytest unit tests will be run automatically before pipeline deployment to prevent against breaking code changes.  To
run your tests locally, run these commands from your anaconda environment shell:

```bash
cd $REPOSITORY_ROOT
pytest
```
