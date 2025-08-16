
$TITLE Dataselection and CVaR model final project
$eolcom //
option optcr=0, reslim=120;

option decimals=6;

* Defining the sets and parameters
Set
    Date   "Weekly time periods"
    AssetName "Names of the assets"


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
    AssetReturn(Date, Asset, AssetName) "Weekly returns"


$gdxin Weekly_returns_2013_2025
$load Date, AssetName, AssetReturn
$gdxin


* Declaring the filtered parameter
Parameter AssetReturn_filtered(Date, Asset, AssetName) "Filtered weekly returns (46)";

* Assigning values only for the 46 selected assets, all dates and all asset names
AssetReturn_filtered(Date, Asset,AssetName) = AssetReturn(Date, Asset, AssetName);


display AssetReturn;



//-------------------------- BOOTSTRAPPING ---------------------------//

* Define sets
Set
    TestPeriod(Date) "Weeks from 2013-01-09 to 2019-08-08"
    scenarios    /s1*s1000/
    weeks        /w1*w4/;

Alias (Date,d);
Alias (scenarios,s);
Alias (weeks,w);

* Restrict to our test period
TestPeriod(d) = yes$(ord(d) <= 344);

display TestPeriod


* Parameters
Parameter
    WeeklyReturn(s,w,Asset)                "Weekly returns drawn for bootstrap"
    MonthlyReturn(s,Asset)                 "Bootstrapped monthly returns"
    ;

Scalar RandNum;

Parameter AssetReturnSimple(Date, Asset);
AssetReturnSimple(Date, Asset) = sum(AssetName, AssetReturn(Date, Asset, AssetName));

display AssetReturnSimple


*Bootstrapping
Loop(s,
    Loop(Asset,
        Loop(w,
            RandNum = uniformint(1, card(TestPeriod));  
            Loop(d$(TestPeriod(d) and ord(d) = RandNum),
                WeeklyReturn(s,w,Asset) = AssetReturnSimple(d,Asset);
            );
        );
    );
);

* Computing monthly compounded returns from selected weeks
Loop(s,
    Loop(Asset,
        MonthlyReturn(s,Asset) = 1;
        Loop(w,
            MonthlyReturn(s,Asset) = MonthlyReturn(s,Asset) * (1 + WeeklyReturn(s,w,Asset));
        );
        MonthlyReturn(s,Asset) = MonthlyReturn(s,Asset) - 1;
    );
);


display MonthlyReturn
* Save filtered and bootstrapped data 
*execute_unload 'filtboot_data.gdx', Asset, Date, AssetName, AssetReturn, WeeklyReturn, MonthlyReturn, TestPeriod;


//-------------------------- IMPLEMENTING CVAR MODEL ---------------------------//



SCALARS
        Budget        'Nominal investment budget'
        alpha         'Confidence level'
;

*Setting budget to 1 mil DKK
Budget = 1000000;
alpha  = 0.95;

VARIABLE 
    VaR
    z
    losses(s)
    CVaR;

POSITIVE VARIABLES
    x(Asset)       'Holdings of assets'
    VaRDev(s)      'Deviation of VaR'
    ;


PARAMETERS      
        pr(s)             'Scenario probability'
        R(s, Asset)       'Final values "R"'
        mu(Asset)         'Expected final values'
;

pr(s) = 1.0 / CARD(s);

R(s, Asset) = 1 + MonthlyReturn(s, Asset);

mu(Asset) = SUM(s, pr(s) * R(s, Asset));



SCALAR
Target_Return
;

Target_Return=(1+0.002589)*Budget;


EQUATIONS
    Budget_cont         'Using the whole budget'
    Loss_Budget         'Losses equal Budget'
    VaRDev_1            'VaRDev greater than losses'
    Target_RetEq        'Equation defining portfolio return vs target'
    
    Object_func_Cvar
    CVaR_cont           'Constraint for CVaR model'
;

Budget_cont .. sum(Asset,x(Asset)) =e= Budget;

Loss_Budget(s) .. losses(s) =e= Budget - sum(Asset, R(s, Asset) * x(Asset));

VaRDev_1(s)   .. VaRDev(s) =g= losses(s) - VaR;

Target_RetEq .. sum(Asset, x(Asset)*mu(Asset)) =g= Target_Return;

Object_func_Cvar .. z=e= CVaR;

CVaR_cont .. CvaR =e= VaR + (sum(s,pr(s)*VaRDev(s)))/(1-alpha)


//------------------------------ MINIMIZING CVAR ---------------------------//


MODEL miCVaRModel /Budget_cont, Loss_Budget, VaRDev_1, Target_RetEq, CVaR_cont, Object_func_Cvar/;

OPTION LP = CPLEX;
OPTION optcr = 0;

scalar PortRet, WorstCase;

ALias(Asset, i);

SOLVE miCVaRModel MINIMIZING z USING LP;
PortRet = SUM(i, mu(i) * x.l(i));
VaR.l = VaR.l;
CVaR.l = CVaR.l;
WorstCase = smax(s, Losses.l(s));
display VaRDev.l, VaR.l, CVaR.l, WorstCase, PortRet, x.l;






//------------------------------ MAXIMIZING EXPECTED RETURN ---------------------------//


SCALAR CVaR_Benchmark 'Maximum acceptable CVaR';

*Calculating monthly benchmark 
CVaR_Benchmark = (0.12/100)*Budget;

VARIABLE
CVAR;

*Add new constraints
EQUATIONS
    ExpRet_Obj  'Objective: maximize expected return'
    CVaR_benchmark_cont 'CVaR must be less than benchmark';
;


ExpRet_Obj .. z =e= sum(Asset, x(Asset)*mu(Asset));  

CVaR_benchmark_cont .. CVaR =l= CVaR_Benchmark;

MODEL MaReturnModel /Budget_cont, Loss_Budget, VaRDev_1, CVaR_cont, ExpRet_Obj, CVaR_benchmark_cont/;

OPTION LP = CPLEX;
OPTION optcr = 0;

Alias(Asset,i)
scalar PortRet, WorstCase;

SOLVE MaReturnModel MAXIMIZING z USING LP;
PortRet = SUM(i, mu(i) * x.l(i));
VaR.l = VaR.l;
CVaR.l = CVaR.l;
WorstCase = smax(s, Losses.l(s));
display VaRDev.l, VaR.l, CVaR.l, WorstCase, PortRet, x.l;










