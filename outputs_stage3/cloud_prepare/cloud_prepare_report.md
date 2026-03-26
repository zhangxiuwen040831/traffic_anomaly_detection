# Stage 3 Cloud Prepare Report

## Remote Target
- Host: qhdlink.lanyun.net:38146
- Remote project root: /root/lanyun-tmp/traffic_anomaly_detection
- Uploaded source bundle: stage3_source_bundle.tar.gz (126405 bytes)

## Executed Steps
- Exit 0: `mkdir -p /root/lanyun-tmp/traffic_anomaly_detection`
- Exit 0: `cd /root/lanyun-tmp/traffic_anomaly_detection && tar -xzf stage3_source_bundle.tar.gz && rm -f stage3_source_bundle.tar.gz`
- Exit 0: `cd /root/lanyun-tmp/traffic_anomaly_detection && chmod +x scripts/cloud/*.sh`
- Exit 0: `cd /root/lanyun-tmp/traffic_anomaly_detection && bash scripts/cloud/stage3_prepare_layout.sh /root/lanyun-tmp/traffic_anomaly_detection`
- Exit 0: `cd /root/lanyun-tmp/traffic_anomaly_detection && bash scripts/cloud/stage3_env_probe.sh outputs_stage3/cloud_prepare/reports`
- Exit 0: `cd /root/lanyun-tmp/traffic_anomaly_detection && find outputs_stage3 -maxdepth 3 -type d | sort`

## Environment Probe Copy
- Local copy: D:\zlx\traffic_anomaly_detection\outputs_stage3\cloud_prepare\remote_environment_probe.txt

## Future GPU Launchers
- `bash scripts/cloud/stage3_download_datasets.sh /root/lanyun-tmp/traffic_anomaly_detection unsw_nb15`
- `bash scripts/cloud/stage3_download_datasets.sh /root/lanyun-tmp/traffic_anomaly_detection cic_ids2017`
- `LAUNCH_MODE=tmux MODEL_NAME=enhanced_mlp_ae DATASET_NAME=unsw_nb15 bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-tmp/traffic_anomaly_detection`
- `LAUNCH_MODE=nohup MODEL_NAME=hybrid_ae DATASET_NAME=cic_ids2017 bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-tmp/traffic_anomaly_detection`