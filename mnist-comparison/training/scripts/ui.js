const statusElem = document.getElementById("status");

function statusLog(status){
    statusElem.innerText = "Status: " + status; 
}

async function triggerStart(){
    await new Promise((resolve) => setTimeout(resolve, 5000));
    // wait for 5 seconds
    let event = new CustomEvent("started");
    console.log("start");
    document.dispatchEvent(event);
}

function triggerEnd(msg){
    let event = new CustomEvent("finished", {
        "detail":{
            message: msg
        }
    });
    console.log(msg);
    document.dispatchEvent(event);
}
