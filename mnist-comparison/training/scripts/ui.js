const statusElem = document.getElementById("status");

function statusLog(status){
    statusElem.innerText = "Status: " + status; 
}

function registerListener(){
    document.addEventListener("started", function(e){ 
        chrome.extension.sendMessage(
            // 发送的e.detailed是一个obj，有两个属性type, message
            e.detailed,
            function(response){
                // callback
            }
        );
    }, false);

    document.addEventListener("finished", function(d){
        chrome.extension.sendMessage( 
            e.detailed,
            function(response){
                // callback
            }
        );
    }, false);
}

function triggerStart(){
    let event = new CustomEvent("started", {
        type: "started",
        message: null
    });
    document.dispatchEvent(event);
}

function triggerEnd(msg){
    let event = new CustomEvent("finished", {
        type: "finished",
        message: msg
    });
    document.dispatchEvent(event);
}