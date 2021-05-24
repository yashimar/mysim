import roslib; roslib.load_manifest('ay_trick')
from ay_py.core.dpl4 import *
from ml_dnn import TNNRegression2


class TModelManager3(TModelManager):
  
  def CreateModels(self, keys=None):
    if keys is None:  keys= [key for key,(In,Out,F) in self.Models.iteritems() if F is None and len(Out)>0]
    if len(keys)==0:  return
    if self.Options['type']=='lwr':
      for key in keys:
        In,Out,F= self.Models[key]
        #dim_in= sum(DimsXSSA(self.SpaceDefs,In))
        #dim_out= sum(DimsXSSA(self.SpaceDefs,Out))
        options= copy.deepcopy(self.Options['lwr_options'])
        options['base_dir']= self.Options['base_dir']
        model= TLWR()
        model.Load(data={'options':options})
        #model.Importance= self.sample_importance  #Share importance in every model
        self.Models[key][2]= model
        self.Learning.update({key})
    elif self.Options['type']=='dnn':
      for key in keys:
        In,Out,F= self.Models[key]
        dim_in= sum(DimsXSSA(self.SpaceDefs,In))
        dim_out= sum(DimsXSSA(self.SpaceDefs,Out))
        options= copy.deepcopy(self.Options['dnn_options'])
        options['base_dir']= self.Options['base_dir']
        options['n_units']= [dim_in] + list(self.Options['dnn_hidden_units']) + [dim_out]
        options['name']= key
        model= TNNRegression2()
        model.Load(data={'options':options})
        self.Models[key][2]= model
        self.Learning.update({key})

def Help():
      pass

def Run(ct, *args):
  print(TModelManager3)