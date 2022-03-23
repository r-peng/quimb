import numpy as np
import time,scipy
from scipy.optimize import optimize
def line_search(f,xk,pk,gk,old_fval,old_old_fval,c1=1e-4):
    phi0 = old_fval
    dphi0 = np.dot(pk,gk)
    print('dphi0=',dphi0)
    def phi(a):
        return f(xk+a*pk)
    a0 = 1.0 
    phi_a0 = phi(a0)
    print('a0={},phi(ai)={}'.format(a0,phi_a0))
    rhs = phi0+c1*a0*dphi0
    if phi_a0<rhs:
        return a0,phi_a0,old_fval,None
    a1 = -dphi0*a0**2/(2.0*(phi_a0-phi0-dphi0*a0))
    phi_a1 = phi(a1)
    print('a1={},phi(ai)={}'.format(a1,phi_a1))
    i = 2
    while phi_a1>=rhs:
        M = np.array([[1.0/a1**2,-1.0/a0**2],[-a0/a1**2,a1/a0**2]])/(a1-a0)
        v = np.array([phi_a1-phi0-dphi0*a1,phi_a0-phi0-dphi0*a0])
        a,b = np.einsum('ij,j->i',M,v)
        a0,a1 = a1,(-b+np.sqrt(b**2-3.0*a*dphi0))/(3.0*a)
        assert a0>0.0 and a1>0.0
        phi_a0,phi_a1 = phi_a1,phi(a1)
        print('a{}={},phi(ai)={}'.format(i,a1,phi_a1))
        i += 1
    return a1,phi_a1,old_fval,None
def line_search_golden(f,xk,pk,gk,old_fval,old_old_fval,eps=1e-3): 
    def _f(a):
        return f(xk+a*pk)
    a1,a4 = 0.0,1.0
    f1,f4 = old_fval,_f(a4)
    phi = (1.0+np.sqrt(5.0))/2.0
    r = phi-1.0
    c = 1.0-r
    a2,a3 = a1+(a4-a1)*c,a1+(a4-a1)*r
    f2,f3 = _f(a2),_f(a3)
    ls = [f1,f2,f3,f4]
    ls.sort()
    i = 0
    while ls[1]-ls[0]>eps:
        print('it=',i)
        print(f1,f2,f3,f4)
        print(a1,a2,a3,a4)
        min12,min34 = min(f1,f2),min(f3,f4)
        if min12<min34:
            a1,a3,a4 = a1,a2,a3
            f1,f3,f4 = f1,f2,f3
            a2 = a1+(a4-a1)*c
            f2 = _f(a2)
        else:
            a1,a2,a4 = a2,a3,a4
            f1,f2,f4 = f2,f3,f4
            a3 = a1+(a4-a1)*r
            f3 = _f(a3)
        ls = [f1,f2,f3,f4]
        ls.sort()
        i += 1
    idx = np.argmin([f1,f2,f3,f4])
    fmin = list([f1,f2,f3,f4])[idx]
    amin = list([a1,a2,a3,a4])[idx]
    return amin,fmin,old_fval,None
############### custom scipy minimize functions ################
def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None, maxiter=None,
                   gtol=1e-5, norm=np.Inf, eps=np.sqrt(np.finfo(float).eps), 
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   hess=None,hessp=None,bounds=None,constraints=(),
                   ls=None,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    from scipy.optimize import optimize
    optimize._check_unknown_options(unknown_options)
    retall = return_all

    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = optimize._prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2 if ls is None else None

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = optimize.vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        if ls is None:
            try:
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                         optimize._line_search_wolfe12(f, myfprime, xk, pk, gfk,
                             old_fval, old_old_fval, amin=1e-100, amax=1e100)
            except optimize._LineSearchError:
                # Line search failed to find a better solution.
                warnflag = 2
                break
        else:
            alpha_k,old_fval,old_old_fval,gfkp1 = \
                ls(xk=xk,pk=pk,gk=gfk,old_fval=old_fval,old_old_fval=old_old_fval)

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        nsk = np.linalg.norm(sk)
        sk /= nsk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        yk /= nsk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = optimize.vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break
        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        print('iter={},rhok_inv={},alpha={},fval={}'.format(k,rhok_inv,alpha_k,old_fval))
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])

    fval = old_fval

    if warnflag == 2:
        msg = optimize._status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = optimize._status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = optimize._status_message['nan']
    else:
        msg = optimize._status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = optimize.OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result
def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=np.sqrt(np.finfo(float).eps), 
                       maxiter=None, disp=False, return_all=False,
                       bounds=None,constraints=(),
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.
    Note that the `jac` parameter (Jacobian) is required.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    optimize._check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0 = np.asarray(x0).flatten()
    # TODO: add hessp (callable or FD) to ScalarFunction?
    sf = optimize._prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )
    f = sf.fun
    fprime = sf.grad
    _h = sf.hess(x0)

    # Logic for hess/hessp
    # - If a callable(hess) is provided, then use that
    # - If hess is a FD_METHOD, or the output fom hess(x) is a LinearOperator
    #   then create a hessp function using those.
    # - If hess is None but you have callable(hessp) then use the hessp.
    # - If hess and hessp are None then approximate hessp using the grad/jac.

    if (hess in optimize.FD_METHODS or isinstance(_h, scipy.sparse.linalg.LinearOperator)):
        fhess = None

        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)

        fhess_p = _hessp

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = optimize.OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
#    while np.add.reduce(np.abs(update)) > xtol:
    b = np.ones(len(x0))
    while np.amax(np.abs(b)) > avextol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.add.reduce(np.abs(b))
        eta = np.min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = np.zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = sf.hess(xk)
            hcalls = hcalls + 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            if np.amax(abs(ri)) < avextol:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = optimize.approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                if isinstance(A, HessianUpdateStrategy):
                    # if hess was supplied as a HessianUpdateStrategy
                    Ap = A.dot(psupi)
                else:
                    Ap = np.dot(A, psupi)
            # check curvature
            Ap = np.asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            print('curv={},npi={},rmax={}'.format(curv/np.dot(psupi,psupi),np.linalg.norm(psupi),np.amax(abs(ri))))
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1          # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     optimize._line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except optimize._LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk = xk + update        # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1
    else:
        if np.isnan(old_fval) or np.isnan(update).any():
            return terminate(3, _status_message['nan'])

        msg = optimize._status_message['success']
        return terminate(0, msg)
def custom(fun,x0,jac=None,callback=None,gtol=1e-5,maxiter=50,t0=0.5,**options):
    print('using scipy.optimize custom')
    fun,grad = fun,jac
    t = t0
    print('t=',t)
    x = x0
    fold = 0.0
    nit = 0
    while nit < maxiter:
        f,g = fun(x),grad(x) 
        if f-fold>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
        x = x - t*g
        if callback is not None:
            callback(x)
        fold = f
        nit += 1
    return scipy.optimize.OptimizeResult(fun=f,jac=g,x=x,nit=nit,message='')
