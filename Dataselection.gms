$TITLE Dataselection final project
* VaR_CVaR.gms: Value at Risk and Conditional Value at Risk models.
$eolcom //
option optcr=0, reslim=120;

option decimals=6;

* Declaring the sets and parameters
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



*AssetReturn(Date, Asset, AssetName)$(not AssetReturn(Date, Asset, AssetName)) = 0;
* replace undefined with 0

*Declaring the filtered parameter
Parameter AssetReturn_filtered(Date, Asset, AssetName) "Filtered weekly returns (46)";

* Assigning values only for the 46 selected assets, all dates and all asset names
AssetReturn_filtered(Date, Asset,AssetName) = AssetReturn(Date, Asset, AssetName);

AssetReturn(Date, Asset, AssetName)$(not AssetReturn(Date, Asset, AssetName)) = 0;

display AssetReturn;



*AssetReturn(Date, Asset, AssetName)= AssetReturn(Date, Asset, AssetName) + 0;
* Save filtered data 
*execute_unload 'filtered_data2.gdx', Asset, Date, AssetName, AssetReturn;


 
*Should have 655*46=30.130 rows. only have 29.799. so missing 331 ????






* Define sets
Set
    TestPeriod(Date) "Weeks from 2013-01-09 to 2019-08-08"
    scenarios    /s1*s1000/
    weeks        /w1*w4/;

Alias (Date,d);
Alias (scenarios,s);
Alias (weeks,w);

* Restrict to our test period

*TestPeriod(Date) = yes$(Date =g= '2013-01-09' and Date =l= '2019-08-08');

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


$exit 











