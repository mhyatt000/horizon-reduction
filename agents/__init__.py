from agents.crl import CRLAgent
from agents.cfgrl import CFGRLAgent
from agents.hcfgrl import HCFGRLAgent
from agents.dsharsa import DSHARSAAgent
from agents.gcfbc import GCFBCAgent
from agents.gcfql import GCFQLAgent
from agents.gciql import GCIQLAgent
from agents.gcsacbc import GCSACBCAgent
from agents.hgcfbc import HGCFBCAgent
from agents.hiql import HIQLAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.sharsa import SHARSAAgent

agents = dict(
    crl=CRLAgent,
    cfgrl=CFGRLAgent,
    hcfgrl=HCFGRLAgent,
    dsharsa=DSHARSAAgent,
    gcfbc=GCFBCAgent,
    gcfql=GCFQLAgent,
    gciql=GCIQLAgent,
    gcsacbc=GCSACBCAgent,
    hgcfbc=HGCFBCAgent,
    hiql=HIQLAgent,
    ngcsacbc=NGCSACBCAgent,
    sharsa=SHARSAAgent,
)
