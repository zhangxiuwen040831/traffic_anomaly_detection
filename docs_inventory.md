# 文档清单与合并方案

## 现有文档清单

| 文件路径 | 所属阶段 | 类型 | 应保留 | 应合并 | 应归档 |
|---------|---------|------|--------|--------|--------|
| PROJECT_OVERVIEW.md | 综合 | 项目概览 | ✅ | ✅ | ❌ |
| README.md | 综合 | 说明文档 | ✅ | ✅ | ❌ |
| README_DEPLOY.md | 第四阶段 | 部署说明 | ✅ | ✅ | ❌ |
| README_FRONTEND.md | 第四阶段 | 前端说明 | ✅ | ✅ | ❌ |
| STAGE2_README.md | 第二阶段 | 训练报告 | ✅ | ✅ | ❌ |
| complete_report.md | 综合 | 综合报告 | ✅ | ✅ | ❌ |
| stage3_future_gpu_plan.md | 第三阶段 | 规划文档 | ✅ | ✅ | ❌ |
| stage3_local_prepare_report.md | 第三阶段 | 准备报告 | ✅ | ✅ | ❌ |
| stage3_local_realdata_summary.md | 第三阶段 | 训练报告 | ✅ | ✅ | ❌ |
| stage5_audit_report.md | 第五阶段 | 审计报告 | ✅ | ✅ | ❌ |
| stage5_defense_report.md | 第五阶段 | 答辩材料 | ✅ | ✅ | ❌ |
| stage5_defense_script_10min.md | 第五阶段 | 答辩材料 | ✅ | ✅ | ❌ |
| stage5_defense_script_5min.md | 第五阶段 | 答辩材料 | ✅ | ✅ | ❌ |
| stage5_ppt_outline.md | 第五阶段 | 答辩材料 | ✅ | ✅ | ❌ |
| stage5_qa_bank.md | 第五阶段 | 答辩材料 | ✅ | ✅ | ❌ |
| stage5_result_trust_decision.md | 第五阶段 | 审计报告 | ✅ | ✅ | ❌ |

## 合并后文档结构

docs/
  00_project_overview.md          # 项目概览
  01_stage_history.md             # 阶段历史
  02_experiment_results.md        # 实验结果
  03_system_deployment.md         # 系统部署
  04_audit_and_trust.md           # 审计与可信度
  05_defense_materials.md         # 答辩材料
  06_limitations_and_future_work.md # 局限性与未来工作
  07_repo_structure.md            # 仓库结构

## 合并说明

1. **00_project_overview.md**：合并 PROJECT_OVERVIEW.md 和 README.md 的项目简介部分
2. **01_stage_history.md**：合并 STAGE2_README.md、stage3_local_prepare_report.md 等阶段文档
3. **02_experiment_results.md**：合并 stage3_local_realdata_summary.md，重点展示第三阶段可信结果
4. **03_system_deployment.md**：合并 README_DEPLOY.md、README_FRONTEND.md
5. **04_audit_and_trust.md**：合并 stage5_audit_report.md、stage5_result_trust_decision.md
6. **05_defense_materials.md**：合并 stage5_defense_report.md、stage5_ppt_outline.md、stage5_qa_bank.md、stage5_defense_script_5min.md、stage5_defense_script_10min.md
7. **06_limitations_and_future_work.md**：合并 stage3_future_gpu_plan.md、complete_report.md 的相关部分
8. **07_repo_structure.md**：新建，说明仓库结构

## 处理原则

1. **保留可信结果**：第三阶段的真实数据实验结果作为主结论
2. **明确工程结果**：第四阶段作为工程实现与系统集成成果展示
3. **移除不可信内容**：第四阶段的满分结果不作为主结论
4. **保持一致性**：确保文档间信息一致，无冲突
5. **结构清晰**：便于读者快速理解项目全貌