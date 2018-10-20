/* jslint node: true */
/* jslint esversion: 6 */
/* globals chrome: false */
/* globals XMLHttpRequest: false */

"use strict";

const defaultSettings = {
  uploadServer: "localhost:8001",
  webpageServer: "localhost:8001",
  firstId: 0, lastId: 30,
  timeout: 600000,
};

chrome.runtime.onInstalled.addListener((details) => {
  chrome.storage.local.set(defaultSettings);
});

function getSettings(entries){
  if (typeof entries === "string") entries = [entries];
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(entries, (result) => {
      if (chrome.runtime.lastError)
        reject(new Error(chrome.runtime.lastError.message));

      resolve(result);
    });
  });
}

function setSettings(entries){
  return new Promise((resolve, reject) => {
    chrome.storage.local.set(entries, () => {
      if (chrome.runtime.lastError)
        reject(new Error(chrome.runtime.lastError.message));

      resolve();
    });
  });
}

function createTab(properties){
  if (properties === void 0)
    properties = {};

  return new Promise((resolve, reject) => {
    chrome.tabs.create(properties, (tab) => {
      resolve(tab);
    });
  });
}

function closeTab(tab){
  return new Promise((resolve, reject) => {
    chrome.tabs.remove(tab.id, () => {
      if (chrome.runtime.lastError)
        reject(new Error(chrome.runtime.lastError.message));

      resolve();
    });
  });
}

function loadUrlInTab(url, tab){
  return new Promise((resolve, reject) => {
    chrome.tabs.update(tab.id, {url}, (tab) => {
      resolve(tab);
    });
  });
}

function webpageEvent(tab, event){
  return new Promise((resolve, reject) => {
    chrome.runtime.onMessage.addListener(function f(message, sender, sendResponse){
      console.log(message);
      if (sender.tab.id !== tab.id) return ;
      const {method, params} = message;
      if (method === event){
        chrome.runtime.onMessage.removeListener(f);
        sendResponse("OK");
        resolve(params);
      }
    });
  });
}

function waiting(timeout){
  return new Promise((resolve, reject) => {
    setTimeout(reject, timeout);
  });
}

class Recording{
  constructor(tab){
    this._tab = tab;
  }

  start(){
    this.title = "";
    this.uploadPs = [];
    this.usages = [];
    this.listeningToProcesses = (processes) => {
      processes = Object.values(processes);
      processes = processes.filter((process) => {
        if (process.type === "extension")
          return false;
        const task = process.tasks[0];
        if (task.tabId !== void 0 && task.tabId !== this._tab.id)
          return false;
        return true;
      });
      //this.usages.push(processes);
      this.uploadPs.push(this.upload(processes));
    };

    chrome.processes.onUpdatedWithMemory.addListener(this.listeningToProcesses);
  }

  async stop(){
    chrome.processes.onUpdatedWithMemory.removeListener(this.listeningToProcesses);
    await Promise.all(this.uploadPs);
    this.uploadPs.length = 0;
  }

  async upload(processes){
    const title = this._title;
    const usage = JSON.stringify(processes);

    const {uploadServer} = await getSettings(["uploadServer"]);
    const url = `http://${uploadServer}/uploadUsage?id=${title}`;
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
    return await new Promise((resolve, reject) => {
      xhr.onreadystatechange = () => {
        resolve();
      };
      xhr.send(`usage=${usage}`);
    });
  }
}

async function uploadMessage(result){
  const title = result.title;
  const message = result.message;

  const {uploadServer} = await getSettings(["uploadServer"]);
  const url = `http://${uploadServer}/uploadMessage?id=${title}`;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
  return await new Promise((resolve, reject) => {
    xhr.onreadystatechange = () => {
      resolve();
    };
    xhr.send(`message=${message}`);
  });
}

async function main(){
  let settings = await getSettings(["firstId", "lastId", "timeout"]);
  const firstId = parseInt(settings.firstId);
  const lastId = parseInt(settings.lastId);
  const timeout = parseInt(settings.timeout);

  for (let i = firstId; i <= lastId; ++i){
    //await setSettings({"firstId": i});
    let tab = await createTab();
    const {webpageServer} = await getSettings("webpageServer");
    const url = `http://${webpageServer}/getWebpage?id=${i}`;
    tab = await loadUrlInTab(url, tab);
    let params;
    params = await webpageEvent(tab, "start");
    const {title} = params;
    const record = new Recording(tab);
    record.title = title;
    record.start();

    params = {};
    try {
      params = await Promise.race([
        webpageEvent(tab, "stop"), waiting(timeout)
      ]);
    } catch (e){
    }

    const {message} = params;
    await record.stop();
    await uploadMessage({title, message});
    await closeTab(tab);
  }

  alert("FINISH");
}

chrome.runtime.onMessage.addListener(function trigger(message, sender, sendResponse){
  const {method, params} = message;
  if (method === "trigger"){
    chrome.runtime.onMessage.removeListener(trigger);
    main();
  }
});
