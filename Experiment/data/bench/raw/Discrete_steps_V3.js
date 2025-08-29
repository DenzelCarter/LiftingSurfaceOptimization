/* //////////////// Discrete steps V2 ////////////////
 The script will sweep between the input values "minVal" and "maxVal". The sweep will be made in discrete "stepsQty" steps. Each step will consist of a settling time "settlingTime" after which a new log entry will be recorded. To reduce noise, "samplesAvg" will be averaged and recorded. This script uses the improved steps2 function, that can introduce a cooling time, as well as a slew-rate limiter for smooth step transitions.
 
The '.' represents a sample is recorded. 5 steps will record 6 data rows (one for zero).

 ^ Motor Input
 |                             __.  maxVal                
 |                         __./   \                 
 |                     __./        \                      
 |                 __./             \           
 |      minVal __./                  \  
 | escInit___./                       \ 
 |_______________________________________> Time
 
///////////// User defined variables //////////// */

var escStart = 1000;     // ESC idle value [700us, 2300us] 
var minVal = 1175;       // Min. input value [700us, 2300us]
var maxVal = 1380;       // Max. input value [700us, 2300us]

// step parameters
var params = {
    steps_qty: 20, // Number of steps
    settlingTime_s: 3, // Settling time before measurement
    cooldownTime_s: 0, // If the motor needs to cool down between steps. Zero disables cooldown.
    cooldownThrottle_us: 1175, // Cool down faster when slowly spinning
    cooldownMinThrottle: 1500, // Only activates the cooldown time for high throttle
    max_slew_rate_us_per_s: 50 // Limits torque from throttle changes
};
   
var samplesAvg = 1;     // Number of samples to average
var repeat = 1; // How many times to repeat the same sequence
var filePrefix = "StepsTestV3";

var rawSamplesPerStep = 1000;      // how many raw rows to log at each step
var rawSampleDelay_s   = 0;        // 0 = as fast as possible

///////////////// Beginning of the script //////////////////

//Start new file
rcb.files.newLogFile({prefix: filePrefix});

//Tare the load cells
rcb.sensors.tareLoadCells(initESC);

//Turn off console spam
rcb.console.setVerbose(false); // mutes internal grey console messages

//Arms the ESC
function initESC(){
    //ESC initialization
    rcb.console.print("Initializing ESC...");
    rcb.output.set("esc", escStart);
    rcb.wait(startSteps, 4);
}

//Start steps
function startSteps(){
    takeSample(ramp);
}

// Records a sample to CSV file
function takeSample(callback){
    rcb.sensors.read(function (result){
        // Write the results and proceed to next step
        rcb.files.newLogEntry(result, callback);
    }, samplesAvg);
}

function collectRaw(n, done){
  if (n <= 0) { done(); return; }
  rcb.sensors.read(function (result){
    rcb.files.newLogEntry(result, function(){
      if (rawSampleDelay_s > 0) rcb.wait(function(){ collectRaw(n-1, done); }, rawSampleDelay_s);
      else collectRaw(n-1, done);
    });
  }, 1); // no averaging
}

// Start the ramp up function
function ramp(){
    rcb.output.steps2("esc", minVal, maxVal, stepFct, finish, params);
}

// The following function will be executed at each step.
function stepFct(nextStepFct){
    collectRaw(rawSamplesPerStep, nextStepFct);  
}

// Ramp back down then finish script
function finish(){
    // Calculate the ramp down time
    var rate = params.max_slew_rate_us_per_s;
    var time = 0;
    if(rate>0){
        time = (maxVal-escStart) / rate;
    }
    rcb.output.ramp("esc", maxVal, escStart, time, endScript);
}

//Ends or loops the script
function endScript() {
    if(--repeat > 0){
      if(repeat === 0){
        rcb.console.overwrite("Running steps…");  // updates one line instead of adding new ones
      }else{
        rcb.console.overwrite("Running steps…");  // updates one line instead of adding new ones
      }
      startSteps();
    }else{
      rcb.endScript();
    }
}