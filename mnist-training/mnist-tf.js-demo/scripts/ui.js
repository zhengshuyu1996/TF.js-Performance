const statusElem = document.getElementById("status");
const infoElem = document.getElementById("info");

function statusLog(status){
    statusElem.innerText = "Status: " + status; 
}

function infoLog(message){
    infoElem.innerText = message + "\n" + infoElem.innerText;
}