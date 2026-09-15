from dataclasses import dataclass
from typing import Dict
@dataclass
class ClusterStats:
    steps:int=0; samples:int=0; spike_events:int=0; tdp_pulses:int=0; cluster_cycles:int=0; if_node_events:Dict[str,int]|None=None
    def __post_init__(self): self.if_node_events={} if self.if_node_events is None else self.if_node_events
    def record_layer(self,name,events,neurons,tdp_bits=5):
        events=int(events); self.if_node_events[name]=self.if_node_events.get(name,0)+events; self.spike_events+=events; self.tdp_pulses+=events*tdp_bits; self.cluster_cycles+=int(neurons)
    def as_dict(self): return self.__dict__.copy()
