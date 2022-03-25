import numpy as np
import time,scipy
from scipy.optimize import optimize
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
