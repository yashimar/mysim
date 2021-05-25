import roslib; roslib.load_manifest('ay_trick')
from ay_py.core.ml_dnn import *
from sklearn.preprocessing import MinMaxScaler


class TNNRegression2(TNNRegression):
  
  def MmsX(self, pred_input = None):
    if pred_input == None:
      if len(self.DataX) == 0:
        return self.DataX
      else:
        mms = MinMaxScaler(feature_range=(-1,1))
        mms.fit(self.DataX)
        return mms.transform(self.DataX)
    else:
      if len(self.DataX) == 0:
        return pred_input
      else:
        mms = MinMaxScaler(feature_range=(-1,1))
        mms.fit(self.DataX)
        return mms.transform([pred_input]).flatten().tolist()
  
      
  def UpdateMain(self):
    if self.NSamples < self.Options['num_min_predictable']:  return

    #Train mean model
    opt={
      'code': '{code}-{n:05d}'.format(n=self.Params['num_train'], code=self.Options['name']+'mean'),
      'log_filename': self.Options['train_log_file'].format(n=self.Params['num_train'], name=self.Options['name'], code='mean', base=self.Options['base_dir']),
      'verb': self.Options['verbose'],
      'gpu': self.Options['gpu'],
      'fwd_loss': self.FwdLoss,
      'optimizer': self.optimizer,
      'x_train': self.MmsX(),
      'y_train': self.DataY,
      'batchsize': self.Options['batchsize'],
      'num_max_update': self.Options['num_max_update'],
      'num_check_stop': self.Options['num_check_stop'],
      'loss_maf_alpha': self.Options['loss_maf_alpha'],
      'loss_stddev_init': self.Options['loss_stddev_init'],
      'loss_stddev_stop': self.Options['loss_stddev_stop'],
      }
    self.TrainNN(**opt)

    # Generate training data for error model
    preds= []
    x_batch= self.MmsX()[:]
    if self.Options['gpu'] >= 0:
      x_batch= cuda.to_gpu(x_batch)
    pred= self.Forward(x_batch, train=False)
    D= self.DataY.shape[1]
    self.DataYErr= np.abs(cuda.to_cpu(pred.data) - self.DataY)

    #Train error model
    opt={
      'code': '{code}-{n:05d}'.format(n=self.Params['num_train'], code=self.Options['name']+'err'),
      'log_filename': self.Options['train_log_file'].format(n=self.Params['num_train'], name=self.Options['name'], code='err', base=self.Options['base_dir']),
      'verb': self.Options['verbose'],
      'gpu': self.Options['gpu'],
      'fwd_loss': self.FwdLossErr,
      'optimizer': self.optimizer_err,
      'x_train': self.MmsX(),
      'y_train': self.DataYErr,
      'batchsize': IfNone(self.Options['batchsize_err'], self.Options['batchsize']),
      'num_max_update': IfNone(self.Options['num_max_update_err'], self.Options['num_max_update']),
      'num_check_stop': IfNone(self.Options['num_check_stop_err'], self.Options['num_check_stop']),
      'loss_maf_alpha': IfNone(self.Options['loss_maf_alpha_err'], self.Options['loss_maf_alpha']),
      'loss_stddev_init': IfNone(self.Options['loss_stddev_init_err'], self.Options['loss_stddev_init']),
      'loss_stddev_stop': IfNone(self.Options['loss_stddev_stop_err'], self.Options['loss_stddev_stop']),
      }
    self.TrainNN(**opt)

    self.Params['num_train']+= 1

    #End of training NNs
    self.is_predictable= True
    
    
  def Predict(self, x, x_var=0.0, with_var=False, with_grad=False):
    res= self.TPredRes()
    x = self.MmsX(x)
    y, y_var, g= self.ForwardX(x, x_var, with_var, with_grad)
    res.Y= y
    res.Var= y_var
    res.Grad= g
    return res

    
def Help():
  pass

def Run(ct, *args):
  print(TNNRegression2())
  