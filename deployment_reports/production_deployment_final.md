# Production Deployment Report

## Deployment Summary
- **Deployment ID**: deploy_1756296030
- **Status**: SUCCESS
- **Environment**: Production
- **Cloud Provider**: KUBERNETES
- **Region**: us-west-2
- **Deployment Time**: 0.00 seconds

## Application Endpoints
- **Web_Ui**: https://darkoperator.terragonlabs.com
- **Api**: https://darkoperator.terragonlabs.com/api/v1
- **Metrics**: https://metrics.darkoperator.terragonlabs.com
- **Logs**: https://logs.darkoperator.terragonlabs.com
- **Status**: https://darkoperator.terragonlabs.com/status

## Monitoring URLs
- **Grafana**: https://grafana.darkoperator.terragonlabs.com
- **Prometheus**: https://prometheus.darkoperator.terragonlabs.com
- **Alertmanager**: https://alerts.darkoperator.terragonlabs.com
- **Kibana**: https://kibana.darkoperator.terragonlabs.com

## Validation Results

### Pre-deployment Validation
- **Passed**: ✅
- **Checks**: 5

### Post-deployment Testing  
- **All Tests Passed**: ✅
- **Pass Rate**: 100.00%

### Final Validation
- **All Checks Passed**: ✅
- **Checks Passed**: 4/4

## Infrastructure
- **Provider**: KUBERNETES
- **Region**: us-west-2
- **Resources Created**: 4

## Monitoring
- **Stack**: Prometheus + Grafana + AlertManager
- **Metrics**: 6 metrics collected
- **Alerts**: 5 alerts configured
- **Dashboards**: 4 dashboards created

## Rollback
If needed, run the following command to rollback:
```
python3 autonomous_production_deployment_final.py --rollback deploy_1756296030
```

## Generated
- **Timestamp**: 2025-08-27T12:00:30.874829
- **Tool**: DarkOperator Studio Autonomous Deployment System v4.0
