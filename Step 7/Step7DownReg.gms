$TITLE Step 7 Backtesting (EDR vs avgRegret target)
$eolcom //
option optcr=0, reslim=120;
option decimals=6;

* ------------------------------------------------------------
* 1) Sets, data, and filtering
* ------------------------------------------------------------
Set
    Date        "Weekly time periods"
    AssetName   "Names of the assets";

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

Parameter AssetReturn_filtered(Date, Asset, AssetName) "Filtered weekly returns (46)";
AssetReturn_filtered(Date, Asset, AssetName) = AssetReturn(Date, Asset, AssetName);

* ------------------------------------------------------------
* 2) Bootstrapping config, caching, and rolling shifts
* ------------------------------------------------------------
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

Scalar RandNum;

Parameter AssetReturnSimple(Date, Asset);
AssetReturnSimple(Date, Asset) = sum(AssetName, AssetReturn(Date, Asset, AssetName));

* Bootstrap settings & cache
option seed=123456;
option limrow=10, limcol=10;

$setglobal RBFILE rolling_bootstrap.gdx
$setglobal REBUILD_RB 0
* set to 1 to force recomputation

Set Shift /sh1*sh1000/;
Set ActiveShift(Shift);

Parameter
    WeeklyReturnRB(s,w,Asset,Shift)   "Bootstrapped weekly returns per shift"
    MonthlyReturnRB(s,Asset,Shift)    "Bootstrapped monthly compounded returns per shift";

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
$load ActiveShift, WeeklyReturnRB=WeeklyReturnRB, MonthlyReturnRB=MonthlyReturnRB
$gdxin
$goto AfterBootstrap

$label DoBootstrap
* -------- Build rolling bootstrap over shifts --------
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

* ------------------------------------------------------------
* 3) Regret target series (avgRegret) and realized rolling
* ------------------------------------------------------------

* Month index aligned to the number of backtest shifts (78 values provided)
Set M /m1*m78/;

Parameter avgRegret(M) /
 m1 0.00403818, m2 0.00399176, m3 0.00399111, m4 0.00398089, m5 0.00393665, m6 0.00389339, m7 0.00385107, m8 0.00380966,
 m9 0.00511435, m10 0.00506051, m11 0.00500780, m12 0.00495617, m13 0.00490560, m14 0.00485605, m15 0.00480749, m16 0.00484538,
 m17 0.00484733, m18 0.00480027, m19 0.00475411, m20 0.00470883, m21 0.00467599, m22 0.00471234, m23 0.00466870, m24 0.00468755,
 m25 0.00464494, m26 0.00460309, m27 0.00456199, m28 0.00452162, m29 0.00460468, m30 0.00456464, m31 0.00452529, m32 0.00449954,
 m33 0.00453250, m34 0.00471474, m35 0.00477790, m36 0.00473842, m37 0.00502002, m38 0.00497920, m39 0.00506577, m40 0.00502525,
 m41 0.00520121, m42 0.00550416, m43 0.00546115, m44 0.00541882, m45 0.00552397, m46 0.00548181, m47 0.00546360, m48 0.00556895,
 m49 0.00552739, m50 0.00548969, m51 0.00544933, m52 0.00542215, m53 0.00538286, m54 0.00536567, m55 0.00538501, m56 0.00541769,
 m57 0.00537953, m58 0.00534192, m59 0.00533482, m60 0.00529803, m61 0.00526174, m62 0.00522595, m63 0.00519063, m64 0.00515580,
 m65 0.00512143, m66 0.00508751, m67 0.00505404, m68 0.00502101, m69 0.00498840, m70 0.00495622, m71 0.00492445, m72 0.00494538,
 m73 0.00491408, m74 0.00506838, m75 0.00525078, m76 0.00521816, m77 0.00518595, m78 0.00515414
/;

Set BacktestShift(Shift);
BacktestShift(Shift) = yes$ActiveShift(Shift);

* Map monthly avgRegret to shifts by ordinal
Parameter TargetRel(Shift) "Monthly target growth used for regret";
TargetRel(Shift) = 0;
TargetRel(Shift)$BacktestShift(Shift) = sum(M$(ord(M)=ord(Shift)), avgRegret(M));

scalar cost /0.001/;        
Scalar tiny /1e-8/;

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

* ------------------------------------------------------------
* 4) Decision variables, trading structure, and EDR model
* ------------------------------------------------------------
Variables z;                           
Positive Variables
    x(Asset)        'Post-rebalance holdings'
    buy(Asset)      'Trade buys'
    sell(Asset)     'Trade sells'
    yminus(s)       'Downside regret per scenario';

Equations
    Budget_cont
    Rebalance
    Target_RetEq
    Regret_Cont
    Object_func_EDR;

Scalar WealthAvail;   WealthAvail = 1000000;     
Scalar TargetGross;

Parameter xPrev(Asset); xPrev(Asset) = 0;

* Budget with transaction costs
Budget_cont .. sum(Asset, x(Asset))
             + cost*sum(Asset, buy(Asset) + sell(Asset)) =e= WealthAvail;

* Rebalancing
Rebalance(Asset) .. x(Asset) =e= xPrev(Asset) + buy(Asset) - sell(Asset);

* Expected (ex-ante) wealth must meet the target
Target_RetEq  .. sum(Asset, muCurr(Asset) * x(Asset)) =g= TargetGross;

* EDR regret constraints and objective (eps=0)
Scalar epsilon /0/;

* yminus(s) >= TargetGross - eps*WealthAvail - SUM_i Rcurr(s,i)*x(i)
Regret_Cont(s) ..
    yminus(s) =G= (TargetGross - epsilon*WealthAvail) - sum(Asset, Rcurr(s,Asset) * x(Asset));

Object_func_EDR ..
    z =E= sum(s, pr(s) * yminus(s));

Model miEDRModel /Budget_cont, Rebalance, Target_RetEq, Regret_Cont, Object_func_EDR/;
option lp = cplex, optcr=0;

* ------------------------------------------------------------
* 5) Results storage
* ------------------------------------------------------------
Parameter
  WeVal(Shift)               'Wealth (ex-post) at end of shift'
  WeAvail(Shift)             'Wealth at start of shift'
  Wts(Asset,Shift)           'Weights after rebalance'
  Holdings(Asset,Shift)      'Holdings (currency) after rebalance'
  ExpectedRegret_path(Shift) 'EDR per shift'
  ExAnteMeanRet(Shift);

* ------------------------------------------------------------
* 6) Backtest loop over shifts (solve EDR, roll forward)
* ------------------------------------------------------------
Loop(Shift$BacktestShift(Shift),


  Rcurr(s,Asset) = 1 + MonthlyReturnRB(s,Asset,Shift);
  muCurr(Asset)  = sum(s, pr(s) * Rcurr(s,Asset));


  TargetGross = (1 + TargetRel(Shift)) * WealthAvail;


  solve miEDRModel minimizing z using lp;

  
  WeAvail(Shift)         = WealthAvail;
  Holdings(Asset,Shift)  = x.l(Asset);
  ExpectedRegret_path(Shift) = z.l;

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

  
  xPrev(Asset) = Rreal(Asset,Shift) * x.l(Asset);
  WealthAvail  = sum(Asset, xPrev(Asset));
  WeVal(Shift) = WealthAvail;

);

* ------------------------------------------------------------
* 7) Outputs for plotting (weights/indices/fan charts)
* ------------------------------------------------------------
Parameter SumWts(Shift), WtsNorm(Asset,Shift);
SumWts(Shift) = sum(Asset, Wts(Asset,Shift));
WtsNorm(Asset,Shift) = 0;
WtsNorm(Asset,Shift)$(SumWts(Shift)>tiny) = Wts(Asset,Shift) / SumWts(Shift);

Parameter WeAvailSafe(Shift), RetExPost(Shift), IndexActual(Shift);
WeAvailSafe(Shift) = max(WeAvail(Shift), tiny);
RetExPost(Shift)   = WeVal(Shift)/WeAvailSafe(Shift) - 1;

Scalar idxA /100/;
Loop(Shift$BacktestShift(Shift),
  idxA = idxA * (1 + RetExPost(Shift));
  IndexActual(Shift) = idxA;
);

* ---- Ex-ante path fan (mean / best / worst) ----
Parameter
    wChosen(Asset,Shift)
  , RcurrAll(s,Asset,Shift)
  , PortGrossScen(s,Shift)
  , IndexScen(s,Shift)
  , TermIndex(s)
  , IndexMean(Shift)
  , IndexBest(Shift)
  , IndexWorst(Shift)
  , IndexTarget(Shift);

RcurrAll(s,Asset,Shift) = 1 + MonthlyReturnRB(s,Asset,Shift);
wChosen(Asset,Shift) = 0;
wChosen(Asset,Shift)$(WeAvail(Shift) > tiny) = Holdings(Asset,Shift) / WeAvail(Shift);

PortGrossScen(s,Shift) = sum(Asset, RcurrAll(s,Asset,Shift) * wChosen(Asset,Shift));

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

Set LastShift(Shift);
Scalar lastOrd;
lastOrd = smax(Shift$BacktestShift(Shift), ord(Shift));
LastShift(Shift) = yes$(ord(Shift) = lastOrd);

TermIndex(s) = sum(Shift$LastShift(Shift), IndexScen(s,Shift));
Set sBest(s), sWorst(s);
sBest(s)  = yes$(TermIndex(s) = smax(ss, TermIndex(ss)));
sWorst(s) = yes$(TermIndex(s) = smin(ss, TermIndex(ss)));

IndexBest(Shift)  = sum(sBest(s),  IndexScen(s,Shift));
IndexWorst(Shift) = sum(sWorst(s), IndexScen(s,Shift));
IndexMean(Shift)  = sum(s, pr(s) * IndexScen(s,Shift));

* Target index built from your avgRegret series
Scalar idxT /100/;
Loop(Shift$BacktestShift(Shift),
  idxT = idxT * (1 + TargetRel(Shift));
  IndexTarget(Shift) = idxT;
);

* ------------------------------------------------------------
* 8) Save to GDX and CSVs (risk → ExpectedRegret; BM → Target)
* ------------------------------------------------------------
execute_unload 'backtest_outputsDR.gdx',
  WtsNorm,
  IndexActual, IndexMean, IndexWorst, IndexBest, IndexTarget,
  ExpectedRegret_path;

* 1) Weights (normalized per shift to sum to 1)
File fW /'weights_longDR.csv'/;  put fW;  put 'Shift,Asset,Weight' /;
Loop(Shift$BacktestShift(Shift),
  Loop(Asset, put Shift.tl:0, ',', Asset.tl:0, ',', WtsNorm(Asset,Shift):12:6 /; );
);
putclose fW;

* 2) Wealth path (start/end of each shift)
File fWe /'wealthDR.csv'/; put fWe; put 'Shift,WealthStart,WealthEnd' /;
Loop(Shift$BacktestShift(Shift),
  put Shift.tl:0, ',', WeAvail(Shift):12:2, ',', WeVal(Shift):12:2 /;
);
putclose fWe;

* 3) Risk path (Expected Downside Regret)
File fR /'riskDR.csv'/; put fR; put 'Shift,ExpectedRegret' /;
Loop(Shift$BacktestShift(Shift),
  put Shift.tl:0, ',', ExpectedRegret_path(Shift):12:6 /;
);
putclose fR;

* 4) Indices (Actual, Ex-ante Mean/Best/Worst, Target)
File fI /'indicesDR.csv'/; put fI;
put 'Shift,IndexActual,IndexMean,IndexWorst,IndexBest,IndexTarget' /;
Loop(Shift$BacktestShift(Shift),
  put Shift.tl:0, ',', IndexActual(Shift):12:4, ',', IndexMean(Shift):12:4, ',',
      IndexWorst(Shift):12:4, ',', IndexBest(Shift):12:4, ',', IndexTarget(Shift):12:4 /;
);
putclose fI;

* 5) Holdings in currency (for stacked area in kr)
File fH /'holdings_longDR.csv'/;  put fH;
put 'Shift,Asset,Holdings' /;
Loop(Shift$BacktestShift(Shift),
  Loop(Asset, put Shift.tl:0, ',', Asset.tl:0, ',', Holdings(Asset,Shift):12:2 /; );
);
putclose fH;

* 6) Monthly scenario returns (Shift × Scenario × Asset)
File fS /'scenarios_monthlyDR.csv'/;  put fS;
put 'Shift,Scenario,Asset,MonthlyReturn' /;
Loop(Shift$BacktestShift(Shift),
  Loop(s,
    Loop(Asset,
      put Shift.tl:0, ',', s.tl:0, ',', Asset.tl:0, ',', MonthlyReturnRB(s,Asset,Shift):12:6 /;
    );
  );
);
putclose fS;
