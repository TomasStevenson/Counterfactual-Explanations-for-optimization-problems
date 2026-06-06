import os, json, numpy as np
os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")
from uc_data_loader import quick_setup
from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b
from uc_decomp_4b import UCDecomp4b
from _decomp_repro_helpers import get_congested_free_lines, build_b_bounds, make_line_weights, UCWeakWCEOracle
DATA_DIR=os.path.join(os.path.dirname(__file__),"Data")
DATA,idx,cvec,b0,u_init,p_init,on_t,off_t=quick_setup(os.path.join(DATA_DIR,"ieee39_newengland.json"),carbon_price=None,voll=20000.0,slack_bus=None)
T=int(DATA.T)
_,solF,_=solve_uc_with_cost_4b(data=DATA,idx=idx,cvec=cvec,window_size=T,per_bus_neutrality=True,u_init=u_init,p_init=p_init,on_time_init=on_t,off_time_init=off_t,output_flag=0)
e=np.array([float(g.emission_rate) for g in DATA.gens]); E=float(np.sum(e[:,None]*solF["p"]))
foil=make_emissions_foil_4b(DATA,alpha=0.10,E_factual=E)
free,util=get_congested_free_lines(solF,b0,thr=0.75); bL,bU=build_b_bounds(b0,free); w=make_line_weights(DATA,b0,util=util)
oracle=UCWeakWCEOracle(data=DATA,cvec=cvec,idx=idx,window_size=T,per_bus_neutrality=True,u_init=u_init,p_init=p_init,on_t=on_t,off_t=off_t,foil_extra_constr_fn=foil,output_flag=0)
b_bs=np.array(json.load(open("bs_39_checkpoint.json"))["best_b"],float)
dec=UCDecomp4b(oracle=oracle,data=DATA,idx=idx,cvec=cvec,foil_extra_constr_fn=foil,b0=b0,b_bounds=(bL,bU),b_free_idx=free,big_M_mu=1e4,verbose=False,w=w,comp_mode="strongdual",b_hat_hint=b_bs,bilinear_exact=True,obbt=True)
vp,_,sp=oracle.solve_plain(b_bs); uk=np.round(sp["u"]).astype(int)
m,mv=dec._build_master_base(T,True,u_init,p_init,on_t,off_t)
dec._add_iteration_block(m,mv,0,uk,u_init,p_init,T,True); m.update()
print("BEFORE OBBT: per free line  b_BS  [bL,bU]")
for ell in free:
    v=m.getVarByName(f"b[{ell}]"); print(f"  ell={ell:2d}  b_BS={b_bs[ell]:.4f}  [{v.LB:.4f},{v.UB:.4f}]")
dec._obbt_root(m,mv,[uk],T,True,u_init,p_init,on_t,off_t)
print("AFTER OBBT:  per free line  b_BS  [LB,UB]  EXCLUDED?")
for ell in free:
    v=m.getVarByName(f"b[{ell}]"); ex = b_bs[ell] < v.LB-1e-9 or b_bs[ell] > v.UB+1e-9
    print(f"  ell={ell:2d}  b_BS={b_bs[ell]:.4f}  [{v.LB:.4f},{v.UB:.4f}]  {'<<< EXCLUDED' if ex else ''}")
