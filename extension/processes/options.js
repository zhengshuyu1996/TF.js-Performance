"use strict";

const usInput = document.getElementById("usInput");
const wsInput = document.getElementById("wsInput");
const fiInput = document.getElementById("fiInput");
const liInput = document.getElementById("liInput");
const sButton = document.getElementById("sButton");

chrome.storage.local.get([
  "uploadServer",
  "webpageServer",
  "firstId", "lastId"
], ({uploadServer, webpageServer, firstId, lastId}) => {
  usInput.value = uploadServer;
  wsInput.value = webpageServer;

  fiInput.value = firstId; liInput.value = lastId;
});

sButton.addEventListener("click", () => {
  chrome.storage.local.set({
    uploadServer: usInput.value,
    webpageServer: wsInput.value,
    firstId: fiInput.value, lastId: liInput.value
  });
});
