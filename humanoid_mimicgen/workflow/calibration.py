"""Identify a local diagonal first-order response from measured command/state pairs.

No empirical integral gain or distance cap. This is local equilibrium compensation,
not a claim of global model validity. Input excitation and prediction error are checked.
"""
import numpy as np
from .contracts import WorkflowError, digest, positive


def identify_response(command_xyz, measured_xyz, *, scope, max_prediction_error_m):
    u,y=np.asarray(command_xyz,float),np.asarray(measured_xyz,float)
    positive(max_prediction_error_m,'prediction acceptance')
    if u.ndim!=2 or u.shape!=y.shape or u.shape[1]!=3 or len(u)<4 or not np.isfinite([u,y]).all():
        raise WorkflowError('needs_calibration','Aligned finite command/state samples required')
    coefficients=[];errors=[]
    for axis in range(3):
        x=np.column_stack([y[:-1,axis],u[:-1,axis],np.ones(len(u)-1)])
        coef,_,rank,_=np.linalg.lstsq(x,y[1:,axis],rcond=None)
        if rank!=3 or not 0<=coef[0]<1 or coef[1]<=0:
            raise WorkflowError('needs_calibration','Unexcited or unstable local response')
        err=float(np.max(abs(x@coef-y[1:,axis])))
        if err>max_prediction_error_m:raise WorkflowError('needs_calibration','Prediction error exceeds declared accuracy')
        coefficients.append(coef.tolist());errors.append(err)
    return {'method':'diagonal_first_order_equilibrium_v1','scope':scope,'coefficients':coefficients,
            'max_prediction_error_m':errors,'acceptance_m':max_prediction_error_m,
            'command_min':u.min(axis=0).tolist(),'command_max':u.max(axis=0).tolist(),
            'sample_count':len(u),'input_digest':digest({'command':u.tolist(),'measured':y.tolist()})}


def equilibrium_command(desired, calibration, scope):
    if calibration.get('method')!='diagonal_first_order_equilibrium_v1' or calibration.get('scope')!=scope:
        raise WorkflowError('needs_calibration','Calibration scope mismatch')
    coef=np.asarray(calibration['coefficients']);a,b,c=coef.T
    command=((1-a)*np.asarray(desired)-c)/b
    if not np.isfinite(command).all() or np.any(command<calibration['command_min']) or np.any(command>calibration['command_max']):
        raise WorkflowError('needs_calibration','Command outside measured calibration envelope')
    return command
