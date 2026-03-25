# Stage 3 Cloud Prepare Report

## Remote Target
- Host: qhdlink.lanyun.net:44410
- Remote project root: /root/lanyun-fs/traffic_anomaly_detection_stage3
- Uploaded source bundle: stage3_source_bundle.tar.gz (109779 bytes)

## Executed Steps
- Exit 0: `mkdir -p /root/lanyun-fs/traffic_anomaly_detection_stage3`
- Exit 0: `cd /root/lanyun-fs/traffic_anomaly_detection_stage3 && tar -xzf stage3_source_bundle.tar.gz && rm -f stage3_source_bundle.tar.gz`
- Exit 0: `cd /root/lanyun-fs/traffic_anomaly_detection_stage3 && chmod +x scripts/cloud/*.sh`
- Exit 0: `cd /root/lanyun-fs/traffic_anomaly_detection_stage3 && bash scripts/cloud/stage3_prepare_layout.sh /root/lanyun-fs/traffic_anomaly_detection_stage3`
- Exit 0: `cd /root/lanyun-fs/traffic_anomaly_detection_stage3 && bash scripts/cloud/stage3_env_probe.sh outputs_stage3/cloud_prepare/reports`
- Exit 0: `cd /root/lanyun-fs/traffic_anomaly_detection_stage3 && find outputs_stage3 -maxdepth 3 -type d | sort`

## Environment Probe Copy
- Local copy: D:\zlx\traffic_anomaly_detection\outputs_stage3\cloud_prepare\remote_environment_probe.txt

## Future GPU Launchers
- `bash scripts/cloud/stage3_download_datasets.sh /root/lanyun-fs/traffic_anomaly_detection_stage3 unsw_nb15`
- `bash scripts/cloud/stage3_download_datasets.sh /root/lanyun-fs/traffic_anomaly_detection_stage3 cic_ids2017`
- `LAUNCH_MODE=tmux MODEL_NAME=enhanced_mlp_ae DATASET_NAME=unsw_nb15 bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-fs/traffic_anomaly_detection_stage3`
- `LAUNCH_MODE=nohup MODEL_NAME=hybrid_ae DATASET_NAME=cic_ids2017 bash scripts/cloud/stage3_gpu_train.sh /root/lanyun-fs/traffic_anomaly_detection_stage3`