from models.cluster_runtime import ClusterStats
def test_accounting():
 s=ClusterStats(steps=64); s.record_layer('if_0',10,100); assert s.tdp_pulses==50 and s.cluster_cycles==100
