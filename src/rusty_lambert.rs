// Dario Izzo Lamb solver
use nalgebra::Vector3;
pub fn lambert_solver(r1: Vector3<f64>, r2: Vector3<f64>, tof: f64, mu: f64) 
-> (Vector3<f64>, Vector3<f64>) {

    let r1_mag=r1.norm();
    let r2_mag=r2.norm();
    let c=r2-r1;
    let c_mag=c.norm();
    let s=(r1_mag+r2_mag+c_mag)/2.0;
    let i=r1.cross(&r2).normalize();
    let lambda=i.z.signum()*(1.0-c_mag/s).sqrt();
    let t=tof*(2.0*mu/s.powi(3)).sqrt();

    let mut x:f64=0.;
    let tol=1e-11;
    let mut its=6900; //could use the smarter solver
    while its!=0 {
        its-=1;
        let t_calc=eval_tof(x,lambda);
        let err=t_calc-t;
        if err.abs()<tol {break;} // found the fkr
        let dtdx=eval_dt(x,lambda);
        let delta_x=err/dtdx;
        x=x-delta_x;
    }

    let y =(1.0-lambda.powi(2)*(1.0-x.powi(2))).max(0.0).sqrt();
    let gamma=(mu*s/2.0).sqrt();
    let rho=(r1_mag-r2_mag)/c_mag;
    let sigma = (1.0 - rho.powi(2)).max(0.0).sqrt();

    let vr1 = gamma * ((lambda * y - x) - rho * (lambda * y + x)) / r1_mag;
    let vt1 = gamma * sigma * (y + lambda * x) / r1_mag;

    let vr2 = -gamma * ((lambda * y - x) + rho * (lambda * y + x)) / r2_mag;
    let vt2 = gamma * sigma * (y + lambda * x) / r2_mag;
 
    //frame unit vec:
    let ir1=r1/r1_mag; let ir2=r2/r2_mag;
    //let it1=ir1.cross(&ir2).cross(&ir1).normalize(); 
    //let it2=ir2.cross(&ir1).cross(&ir2).normalize();
    let it1 = i.cross(&ir1); 
    let it2 = i.cross(&ir2);
    let v1=vr1*ir1+vt1*it1; let v2=vr2*ir2+vt2*it2;

    // let mut vi=Vector3::new(0.,0.,0.);
    // let mut vf=Vector3::new(0.,0.,0.); 
    // for i in 0..3 {
    //     vi[i]=v1[i];
    //     vf[i]=v2[i];
    // }

    (v1, v2)
}

//helper functions
fn eval_tof(x: f64, lambda: f64) -> f64 {
    let z=1.0-x*x;
    if z.abs()<1e-5{
        let d=1e-4;
        let t_minus=eval_tof_exact(1.0-d,lambda);
        let t_plus=eval_tof_exact(1.0+d,lambda);
        let slope=(t_plus-t_minus)/(2.0*d);
        let t_1=2.0/3.0*(1.0-lambda.powi(3));
        return t_1+slope*(x-1.0);
    }
    eval_tof_exact(x,lambda)
}
fn eval_tof_exact(x: f64, lambda: f64) -> f64 {
    let z=1.0-x*x;
    let y=(1.0-lambda*lambda*z).max(0.0).sqrt();
    if x < 1.  {
        let arg=(x*y+lambda*z).clamp(-1.0,1.0);
        (arg.acos()/z.sqrt()-x+lambda*y)/z
    } else {
        let arg=(x*y-lambda*(-z)).max(1.0);
        (x-lambda*y-arg.acosh()/(-z).sqrt())/(-z)
    }
}
fn eval_dt(x: f64, lambda: f64) -> f64 {
    let z=1.-x*x;
    if z.abs()<1e-4 {
        let h=1e-5;
        return (eval_tof(x+h,lambda)-eval_tof(x-h,lambda))/(2.0*h);}
    let y=(1.0-lambda*lambda*z).max(0.0).sqrt();
    let t=eval_tof_exact(x,lambda);
    (3.*x*t-2.+2.*lambda.powi(3)*x/y)/z
}

