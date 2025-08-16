$TITLE Step 7 Backtesting
* VaR_CVaR.gms: Value at Risk and Conditional Value at Risk models.
$eolcom //
option optcr=0, reslim=120;

option decimals=6;

* Declaring the sets and parameters
Set
    Date   "Weekly time periods"
    AssetName "Names of the assets";


* Subset of chosen assets
Set Asset "46 selected assets"
/ LU0589470672
  LU0376447578
  LU0376446257
  LU0471299072
  LU1893597564
  DK0060189041
  DK0010264456
  DK0060051282
  LU0332084994
  DK0016023229
  DK0061553245
  DK0016261910
  DK0016262728
  DK0016262058
  DK0010106111
  DK0060240687
  DK0060005254
  DK0010301324
  DK0060300929
  DK0060158160
  LU1028171921
  LU0230817339
  DK0060815090
  DK0061067220
  DK0060498509
  DK0060498269
  DK0061134194
  DK0016300403
  DK0061150984
  IE00B02KXK85
  DE000A0Q4R36
  IE00B27YCF74
  IE00B1XNHC34
  DE000A0H08H3
  DE000A0Q4R28
  IE00B0M63516
  IE00B0M63623
  IE00B5WHFQ43
  DE000A0H08S0
  IE00B42Z5J44
  DE000A0H08Q4
  IE00B5377D42
  IE00B0M63391
  IE00B2QWCY14
  DK0061544178
  DK0061544418
/;


Parameter
    AssetReturn(Date, Asset, AssetName) "Weekly returns";


$gdxin Weekly_returns_2013_2025
$load Date, AssetName, AssetReturn
$gdxin


*Declaring the filtered parameter
Parameter AssetReturn_filtered(Date, Asset, AssetName) "Filtered weekly returns (46)";

* Assigning values only for the 46 selected assets, all dates and all asset names
AssetReturn_filtered(Date, Asset,AssetName) = AssetReturn(Date, Asset, AssetName);
*display AssetReturn;



//-------------------------- BOOTSTRAPPING ---------------------------//
* Define sets
Set
    TestPeriod(Date) "Weeks from 2013-01-09 to 2019-08-08"
    scenarios    /s1*s1000/
    weeks        /w1*w4/;

Alias (Date,d);
Alias (scenarios,s);
Alias (weeks,w);
Alias (s, ss);


* Restrict to our test period
TestPeriod(d) = yes$(ord(d) <= 344);

*TestPeriod(Date) = yes$(Date =g= '2013-01-09' and Date =l= '2019-08-08');
display TestPeriod



Scalar RandNum;

Parameter AssetReturnSimple(Date, Asset);
AssetReturnSimple(Date, Asset) = sum(AssetName, AssetReturn(Date, Asset, AssetName));

*display AssetReturnSimple

* -------------------- BOOTSTRAP CONFIG & CACHING --------------------
option seed=123456;
option limrow=10, limcol=10;

$setglobal RBFILE rolling_bootstrap.gdx
$setglobal REBUILD_RB 0
* set to 1 to force recompute & overwrite cache

* If you already declared these sets above, keep them there and DO NOT redeclare.
* We keep your TestPeriod/scenarios/weeks/Alias/RandNum/AssetReturnSimple from above.

* Containers for caching (new names to avoid clashes)
Set Shift /sh1*sh1000/;
Set ActiveShift(Shift);

Parameter
    WeeklyReturnRB(s,w,Asset,Shift)   "Bootstrapped weekly returns per shift"
    MonthlyReturnRB(s,Asset,Shift)    "Bootstrapped monthly compounded returns per shift";
    


* --- Shared bootstrap/backtest scalars (must exist even when we load from cache)
Scalar
    step     /4/                
  , trainLen                     
  , totalDates                   
  , nShifts                      
  , startIdx, endIdx;            

trainLen   = card(TestPeriod);
totalDates = card(Date);


$ifthen %REBUILD_RB%==1
$goto DoBootstrap
$endif

$if not exist %RBFILE% $goto DoBootstrap

* -------- Load cached rolling bootstrap --------
$gdxin %RBFILE%
$load ActiveShift, WeeklyReturnRB, MonthlyReturnRB
$gdxin
$goto AfterBootstrap


$label DoBootstrap
* -------------------- ROLLING BOOTSTRAP (4-week shifts) --------------------




trainLen   = card(TestPeriod);
totalDates = card(Date);
nShifts    = 1 + floor( (totalDates - trainLen) / step );

ActiveShift(Shift) = yes$(ord(Shift) <= nShifts);



Loop(Shift$(ActiveShift(Shift)),
    startIdx = 1 + (ord(Shift)-1)*step;
    endIdx   = startIdx + trainLen - 1;
    if (endIdx > totalDates, endIdx = totalDates);

    Loop(s,
        Loop(Asset,
            Loop(w,
                RandNum = uniformint(startIdx, endIdx);
                Loop(d$(ord(d) = RandNum),
                    WeeklyReturnRB(s,w,Asset,Shift) = AssetReturnSimple(d,Asset);
                );
            );
            MonthlyReturnRB(s,Asset,Shift) = 1;
            Loop(w,
                MonthlyReturnRB(s,Asset,Shift) = MonthlyReturnRB(s,Asset,Shift) * (1 + WeeklyReturnRB(s,w,Asset,Shift));
            );
            MonthlyReturnRB(s,Asset,Shift) = MonthlyReturnRB(s,Asset,Shift) - 1;
        );
    );
);

execute_unload '%RBFILE%', ActiveShift, WeeklyReturnRB, MonthlyReturnRB;

$label AfterBootstrap

* Small peek (avoid huge listing); remove if not needed

*Scalar nShift, nScen, nAsset;
*nShift = card(Shift);  nScen = card(scenarios);  nAsset = card(Asset);
*display nShift, nScen, nAsset;

*Parameter Peek(s,Asset,Shift);
*Peek(s,Asset,Shift) = MonthlyReturnRB(s,Asset,Shift);

*Parameter PeekSmall(s,Asset,Shift);
*PeekSmall(s,Asset,Shift)$(ord(s)=1 and ord(Asset)<=5 and ord(Shift)<=5) = Peek(s,Asset,Shift);
*display PeekSmall;




* ===================== BACKTEST: CVaR with 4-week revisions =====================

scalar alpha /0.95/;          
scalar cost   /0.001/;        
Scalar tiny /1e-8/; 


Set BacktestShift(Shift);
BacktestShift(Shift) = yes$ActiveShift(Shift);

* ----- Benchmark targets per shift (given in PERCENT; convert to decimals) -----
Set I /i1*i78/;   

Parameter BMretPct(I) /
  i1 0.258859, i2 0.266308, i3 0.260682, i4 0.256140, i5 0.256957, i6 0.260036, i7 0.265951, i8 0.271726,
  i9 0.130058, i10 0.174897, i11 0.183291, i12 0.203506,
  i13 0.204395, i14 0.206372, i15 0.213714, i16 0.204706, i17 0.199478, i18 0.223336, i19 0.223490, i20 0.234815,
  i21 0.233012, i22 0.224413, i23 0.230159, i24 0.223396, i25 0.230722, i26 0.239167, i27 0.239885, i28 0.239413,
  i29 0.226422, i30 0.233250, i31 0.235543, i32 0.233694, i33 0.226018, i34 0.203289, i35 0.192774, i36 0.192640,
  i37 0.159975, i38 0.162734, i39 0.150524, i40 0.172783, i41 0.151041, i42 0.116304, i43 0.123596, i44 0.141251,
  i45 0.126760, i46 0.143230, i47 0.141090, i48 0.126554, i49 0.133194, i50 0.133119, i51 0.138487, i52 0.137435,
  i53 0.141319, i54 0.139353, i55 0.133776, i56 0.126952, i57 0.142534, i58 0.159879, i59 0.156914, i60 0.166068,
  i61 0.171024, i62 0.171653, i63 0.171711, i64 0.172652, i65 0.177478, i66 0.183254, i67 0.184593, i68 0.189469,
  i69 0.193529, i70 0.195427, i71 0.201402, i72 0.195931, i73 0.201059, i74 0.182177, i75 0.160791, i76 0.183036,
  i77 0.189038, i78 0.188951
/;


Parameter BMret(Shift);              

BMret(Shift) = 0;
BMret(Shift)$BacktestShift(Shift) = sum(I$(ord(I)=ord(Shift)), BMretPct(I)/100);

*display BMret;



Parameter pr(s);  pr(s) = 1/card(s);

Parameter Rcurr(s,Asset), muCurr(Asset);

* Realized (ex-post) 4-week gross for each shift using actual data
Parameter Rreal(Asset,Shift);
Loop(Shift$BacktestShift(Shift),
  startIdx = 1 + (ord(Shift)-1)*4;               
  endIdx   = startIdx + trainLen - 1;
  Rreal(Asset,Shift) = 1;
  Loop(weeks,
    Loop(Date$(ord(Date) = endIdx + ord(weeks)),
      Rreal(Asset,Shift) = Rreal(Asset,Shift) * (1 + AssetReturnSimple(Date,Asset));
    );
  );
);

* -------- CVaR model (reused every shift) --------
Variables VaR, z, CVaR, losses(s);
Positive Variables
    x(Asset)        'Post-rebalance holdings'
    VaRDev(s)       'Excess loss over VaR'
    buy(Asset)      'Trade buys'
    sell(Asset)     'Trade sells';

Equations
    Budget_cont
    Rebalance
    Loss_Budget
    VaRDev_1
    Target_RetEq
    Object_func_Cvar
    CVaR_cont;

Scalar WealthAvail;   WealthAvail = 1000000;     
Scalar TargetGross;

Parameter xPrev(Asset); xPrev(Asset) = 0;

Budget_cont .. sum(Asset, x(Asset))
             + cost*sum(Asset, buy(Asset) + sell(Asset)) =e= WealthAvail;

Rebalance(Asset) .. x(Asset) =e= xPrev(Asset) + buy(Asset) - sell(Asset);

Loss_Budget(s) .. losses(s) =e= WealthAvail - sum(Asset, Rcurr(s,Asset) * x(Asset));

VaRDev_1(s)   .. VaRDev(s)  =g= losses(s) - VaR;

Target_RetEq  .. sum(Asset, muCurr(Asset) * x(Asset)) =g= TargetGross;

Object_func_Cvar .. z =e= CVaR;

CVaR_cont .. CvaR =e= VaR + (sum(s, pr(s) * VaRDev(s))) / (1 - alpha);

Model miCVaRModel /Budget_cont, Rebalance, Loss_Budget, VaRDev_1, Target_RetEq, CVaR_cont, Object_func_Cvar/;
option lp = cplex, optcr=0;

* -------- Results storage --------
Parameter
  WeVal(Shift)               'Wealth (ex-post) at end of shift'
  WeAvail(Shift)             'Wealth at start of shift'
  Wts(Asset,Shift)           'Weights after rebalance'
  Holdings(Asset,Shift)      'Holdings (currency) after rebalance'
  VaR_path(Shift)
  CVaR_path(Shift)
  ExAnteMeanRet(Shift);

* -------- Loop over shifts --------
Loop(Shift$BacktestShift(Shift),

  Rcurr(s,Asset) = 1 + MonthlyReturnRB(s,Asset,Shift);
  muCurr(Asset)  = sum(s, pr(s) * Rcurr(s,Asset));

  TargetGross = (1 + BMret(Shift)) * WealthAvail;

  solve miCVaRModel minimizing z using lp;

  
WeAvail(Shift)         = WealthAvail;
Holdings(Asset,Shift)  = x.l(Asset);
VaR_path(Shift)        = VaR.l;
CVaR_path(Shift)       = CVaR.l;

if (WealthAvail <= tiny,
    
    Wts(Asset,Shift)      = 0;
    ExAnteMeanRet(Shift)  = 0;
else
    Wts(Asset,Shift)      = x.l(Asset) / WealthAvail;
    ExAnteMeanRet(Shift)  = (sum(Asset, muCurr(Asset) * x.l(Asset)) / WealthAvail) - 1;
);

if (WealthAvail <= tiny,
   put_utility 'log' / 'WARNING: WealthAvail ~ 0 at shift ' Shift.tl:0;
);

* Roll portfolio forward to next shift (realized ex-post)
xPrev(Asset) = Rreal(Asset,Shift) * x.l(Asset);
WealthAvail  = sum(Asset, xPrev(Asset));
WeVal(Shift) = WealthAvail;


);

*option limrow=10, limcol=10;
*display WeAvail, WeVal, VaR_path, CVaR_path;

*Parameter WtsPeek(Asset,Shift);
*WtsPeek(Asset,Shift)$(ord(Asset)<=10 and ord(Shift)<=5) = Wts(Asset,Shift);
*display WtsPeek;



* ========= Build outputs for plotting (safe) =========

* 7.6.1 Stacked weights (normalize to 100% per shift), avoid /0
Parameter SumWts(Shift), WtsNorm(Asset,Shift);
SumWts(Shift) = sum(Asset, Wts(Asset,Shift));
WtsNorm(Asset,Shift) = 0;
WtsNorm(Asset,Shift)$(SumWts(Shift)>tiny) = Wts(Asset,Shift) / SumWts(Shift);

* 7.6.2 Actual 4-week return per shift and wealth index (start at 100), avoid /0
Parameter WeAvailSafe(Shift), RetExPost(Shift), IndexActual(Shift);
WeAvailSafe(Shift) = max(WeAvail(Shift), tiny);
RetExPost(Shift)   = WeVal(Shift)/WeAvailSafe(Shift) - 1;

Scalar idxA /100/;
Loop(Shift$BacktestShift(Shift),
  idxA = idxA * (1 + RetExPost(Shift));
  IndexActual(Shift) = idxA;
);



*-----------------------------------------------------------------------
* EX-ANTE LINES (PATH-BASED): best/worst are single scenario paths.
*-----------------------------------------------------------------------
Parameter
    wChosen(Asset,Shift)
  , RcurrAll(s,Asset,Shift)
  , PortGrossScen(s,Shift)
  , IndexScen(s,Shift)
  , TermIndex(s)
  , IndexMean(Shift)
  , IndexBest(Shift)
  , IndexWorst(Shift)
  , IndexBM(Shift);

* Scenario gross per asset (from bootstrap)
RcurrAll(s,Asset,Shift) = 1 + MonthlyReturnRB(s,Asset,Shift);

* Chosen weights from optimized holdings (currency -> weights)
wChosen(Asset,Shift) = 0;
wChosen(Asset,Shift)$(WeAvail(Shift) > tiny) = Holdings(Asset,Shift) / WeAvail(Shift);

* Portfolio gross per scenario & shift given fixed weights  <-- (the missing part!)
PortGrossScen(s,Shift) = sum(Asset, RcurrAll(s,Asset,Shift) * wChosen(Asset,Shift));

* Build a cumulative index for each scenario path (start at 100)
IndexScen(s,Shift) = 0;
Scalar idx0 /100/;
Scalar idxS;  

Loop(s,
  idxS = idx0;
  Loop(Shift$BacktestShift(Shift),
    idxS = idxS * PortGrossScen(s,Shift);
    IndexScen(s,Shift) = idxS;
  );
);

* Identify last backtest shift
Set LastShift(Shift);
Scalar lastOrd;
lastOrd = smax(Shift$BacktestShift(Shift), ord(Shift));
LastShift(Shift) = yes$(ord(Shift) = lastOrd);

* Terminal index per scenario; pick best/worst PATHS (use alias ss in smax/smin)
TermIndex(s) = sum(Shift$LastShift(Shift), IndexScen(s,Shift));
Set sBest(s), sWorst(s);
sBest(s)  = yes$(TermIndex(s) = smax(ss, TermIndex(ss)));
sWorst(s) = yes$(TermIndex(s) = smin(ss, TermIndex(ss)));

* Exported ex-ante indices
IndexBest(Shift)  = sum(sBest(s),  IndexScen(s,Shift));
IndexWorst(Shift) = sum(sWorst(s), IndexScen(s,Shift));
IndexMean(Shift)  = sum(s, pr(s) * IndexScen(s,Shift));  

* Benchmark index (compound BMret per shift)
Scalar idxBM /100/;
Loop(Shift$BacktestShift(Shift),
  idxBM = idxBM * (1 + BMret(Shift));
  IndexBM(Shift) = idxBM;
);







* ---- Save to GDX for Python plotting ----
execute_unload 'backtest_outputs.gdx',
  WtsNorm,
  IndexActual, IndexMean, IndexWorst, IndexBest, IndexBM,
  VaR_path, CVaR_path;

* ---- CSVs ----
* 1) Weights (normalized per shift to sum to 1)
File fW /'weights_long.csv'/;  put fW;  put 'Shift,Asset,Weight' /;
Loop(Shift$BacktestShift(Shift),
  Loop(Asset, put Shift.tl:0, ',', Asset.tl:0, ',', WtsNorm(Asset,Shift):12:6 /; );
);
putclose fW;

* 2) Wealth path (start/end of each shift)
File fWe /'wealth.csv'/; put fWe; put 'Shift,WealthStart,WealthEnd' /;
Loop(Shift$BacktestShift(Shift),
  put Shift.tl:0, ',', WeAvail(Shift):12:2, ',', WeVal(Shift):12:2 /;
);
putclose fWe;

* 3) Risk path (VaR & CVaR)
File fR /'risk.csv'/; put fR; put 'Shift,VaR,CVaR' /;
Loop(Shift$BacktestShift(Shift),
  put Shift.tl:0, ',', VaR_path(Shift):12:6, ',', CVaR_path(Shift):12:6 /;
);
putclose fR;

* 4) Indices (Actual, Ex-ante Mean/Best/Worst, Benchmark)
File fI /'indices.csv'/; put fI;
put 'Shift,IndexActual,IndexMean,IndexWorst,IndexBest,IndexBM' /;
Loop(Shift$BacktestShift(Shift),
  put Shift.tl:0, ',', IndexActual(Shift):12:4, ',', IndexMean(Shift):12:4, ',',
      IndexWorst(Shift):12:4, ',', IndexBest(Shift):12:4, ',', IndexBM(Shift):12:4 /;
);
putclose fI;

* 5) Holdings in currency (for stacked area in kr)
File fH /'holdings_long.csv'/;  put fH;
put 'Shift,Asset,Holdings' /;
Loop(Shift$BacktestShift(Shift),
  Loop(Asset, put Shift.tl:0, ',', Asset.tl:0, ',', Holdings(Asset,Shift):12:2 /; );
);
putclose fH;


* 6) Monthly scenario returns (one row per Shift × Scenario × Asset)
*    NOTE: This can be large: card(Shift)*card(s)*card(Asset) rows.
File fS /'scenarios_monthly.csv'/;  put fS;
put 'Shift,Scenario,Asset,MonthlyReturn' /;
Loop(Shift$BacktestShift(Shift),
  Loop(s,
    Loop(Asset,
      put Shift.tl:0, ',', s.tl:0, ',', Asset.tl:0, ',', MonthlyReturnRB(s,Asset,Shift):12:6 /;
    );
  );
);
putclose fS;
