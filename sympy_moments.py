import sympy as sm
x = sm.Symbol('x',real=True)

def L():
    return s**2/(2*T**2)

def u():
    return L()/(1+T) - sm.log(rho)/T

u,L,T,rho,rho0,s = sm.var('u,L,T,rho,rho0,s',real=True,positive=True)

def lognormal(x):
    """
    Returns P(x=log y) 
    """
    return 1/sm.sqrt(2*sm.pi*s**2) * sm.exp(-(x+s**2/2)**2 / (2*s**2))

def besselgaussian(x):
    return sm.besseli(1,x) * sm.exp(-(x/2)**2 / L) 

def hopkins(u):
    return sm.besseli(1,x) * sm.exp(-(x/2)**2 / L) 

def rho(x):
    return rho0*sm.exp(T*(x**2/(4*L)+L/(1+T)))

def logrho(x):
    return sm.log(rho0) + T*(x**2/(4*L)+L/(1+T))

def expectation(var, intvar, function=besselgaussian):
    expr = var * besselgaussian(intvar)
    return sm.simplify( sm.integrate(expr,(intvar,0,sm.oo), conds='none') )

if __name__ == "__main__":
    x = sm.Symbol('x',real=True)
    print "Hopkins PDF: "
    print "E[rho] = ",expectation(rho(x),x)
    print "E[rho^2] = ",expectation(rho(x)**2,x)
    print "E[rho^3] = ",expectation(rho(x)**3,x)
    import sys
    sys.setrecursionlimit(5000)
    logrho = sm.simplify(logrho())
    print "E[log(rho)] = ",expectation(logrho,x)
    print "E[log(rho)^2] = ",expectation(logrho**2,x)
    print "E[rho log(rho)] = ",expectation(rho(x)*logrho,x)
    print "E[rho log(rho)^2] = ",expectation(rho(x)*logrho**2,x)

    T,s = sm.var('T,s',real=True,positive=True)
    sm.solve(T - 0.1 * s**2 * (1+T)**-3, T)
