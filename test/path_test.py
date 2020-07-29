#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.sm4')
from tsim.sm4 import SetMaterial
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *

def Run(ct,*args):
  # Print(TGraphDynDomain)
  # Print(TLocalLinear)
  # Print(OpenW)
  Print(TGraphDynPlanLearn.MM.Update)