
$TITLE Dataselection and downside regret model final project

$eolcom //
option optcr=0, reslim=120;

option decimals=6;

* Defining the sets and parameters
Set
    Date   "Weekly time periods"
    AssetName "Names of the assets"


* Subset of 46 chosen assets
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


*Declaring the filtered parameter
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


*TestPeriod(Date) = yes$(Date =g= '2013-01-09' and Date =l= '2019-08-08');
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


* Bootstrapping
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

* Computing the monthly compounded returns from selected weeks
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



//-------------------------- IMPLEMENTING DOWNSIDE REGRET MODEL ---------------------------//


Alias(Asset,i)
Alias(Date,d);


PARAMETER
    pr(s)                   "Probability of scenario s"
    Rbar(s,i)               "Expected price or return for asset i at scenario ss"
    g(s)                    "Target value in scenario s"
    epsilon                 "Epsilon threshold"
    P0(i)                   "Initial price of asset i"
    mu                      "Target return rate"
    V0                      "Initial budget"
    Pbar(i)                 "Expected monthly return for asset i (expected return at horizon)";
    ;
    
*Initial budget set to 1 million DKK
V0=1000000;

*Setting target return to be 2% a year (0.1651% monthly)
mu=(0.258859/100)+1;
g(s)=((0.1651/100)+1)*V0;

*Setting epsilon to zero
epsilon=0.00;


VARIABLES
    z               "Objective value"
    ;

POSITIVE VARIABLES
    x(i)            "Weight of asset i"
    yminus(s)       "Downside regret in scenario l"
    ;



pr(s) = 1.0 / CARD(s);

Rbar(s, i) = 1 + MonthlyReturn(s, i);

Pbar(i) = 1+sum(s, pr(s) * MonthlyReturn(s,i));

*display Rbar, Pbar;

P0(i)=1


EQUATIONS
    expReturn   "Expected return constraint"
    regret_cont(s) "Regret definition per scenario"
    budget      "Budget constraint"
    objdef      "Objective function definition"
    ;


expReturn.. SUM(i, Pbar(i) * x(i)) =G= mu * V0;

regret_cont(s).. yminus(s) =G= (g(s) - eps * V0) - SUM(i,Rbar(s,i) * x(i));

budget.. SUM(i, P0(i) * x(i)) =E= V0;

objdef.. z =E= SUM(s, pr(s) * yminus(s));


//------------------------ MINIMIZING DOWNSIDE REGRET ----------------------------//


MODEL downside_regret /expReturn, regret_cont, budget, objdef/;

OPTION LP = CPLEX;
OPTION optcr = 0;

scalar PortRet;



SOLVE downside_regret MINIMIZING z USING LP;

PortRet = SUM(i, Pbar(i) * x.l(i));
display x.l, yminus.l, PortRet, z.l;


//------------------------ MAXIMIZING EXPECTED RETURN ----------------------------//

PARAMETER omega;

omega=0.004038*V0

EQUATIONS
    object_maxi      "Objective function dwhen maximizing return"
    dr_constraint   "Constraint regarding downside regret"
    ;

object_maxi .. z =E= SUM(i, Pbar(i) * x(i)) ;

dr_constraint .. SUM(s, pr(s) * yminus(s)) =L= omega 


MODEL ma_return_dr /dr_constraint, regret_cont, budget, object_maxi/;

OPTION LP = CPLEX;
OPTION optcr = 0;

scalar PortRet
        DownsideReg;


SOLVE ma_return_dr MAXIMIZING z USING LP;

PortRet = SUM(i, Pbar(i) * x.l(i));
DownsideReg=SUM(s, pr(s) * yminus.l(s));
display x.l, yminus.l, PortRet, z.l, DownsideReg
;





