"""Functional state-RAM restore remap; no timing or energy model."""
import torch


class MacroStateRAM:
    def __init__(self, template):
        self.mr=torch.zeros_like(template)
        self.scu=torch.zeros_like(template)
        self.records_read=self.records_written=0
        self.remapped_records=0

    def restore(self, exchange=False, active_macros=16):
        if active_macros not in (8,16):raise ValueError('expected 8 or 16')
        # Each destination reads exactly one source record. Weight/input
        # addresses do not participate in this mapping.
        address=torch.arange(16,device=self.mr.device)
        if exchange:
            address[:active_macros]=(address[:active_macros]+active_macros//2)%active_macros
            self.remapped_records+=self.mr.numel()//(16*5)*active_macros
        self.records_read+=self.mr.numel()//5
        return self.mr.index_select(-2,address),self.scu.index_select(-2,address)

    def writeback(self,mr,scu):
        # Slots are indexed by physical Macro after this step. Thus remap
        # is enabled only on an exchange boundary, not on every later read.
        self.mr=mr.clone();self.scu=scu.clone()
        self.records_written+=mr.numel()//5

    def report(self):
        return dict(macro_state_records_read=self.records_read,
                    macro_state_records_written=self.records_written,
                    remapped_read_records=self.remapped_records)
