const TEST_SIZE = 10;
const DATA_SIZE = 10;
const picHolder = document.getElementById("pic holder");
for (let i = 1; i <= 10; i++){
    let elem = document.createElement("canvas");
    elem.setAttribute("id", "pic"+i);
    elem.setAttribute("width", 224);
    elem.setAttribute("height", 224);
    let img = new Image();
    img.crossOrigin = "Anonymous"; // important
    img.onload = function(){
        elem.getContext("2d").drawImage(img, 0, 0);
    }
    img.src =  LOCAL_SERVER+"/data/224/pic"+i+".png";


    picHolder.appendChild(elem);
}