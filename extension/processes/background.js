/* jslint node: true */
/* jslint esversion: 6 */
/* globals chrome: false */
/* globals XMLHttpRequest: false */

"use strict";

const defaultSettings = {
  uploadServer: "localhost:8001",
  webpageServer: "localhost:8001",
  firstId: 1, lastId: 10,
  timeout: 600000
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

  static _startMonitoringProcess(tab){
    if (Recording.count === void 0){
      Recording.count = 0;
      Recording.usages = [];
      Recording.listeningToProcesses = (processes) => {
        processes = Object.values(processes);
        processes = processes.filter((process) => {
          return process.type !== "extension";
        });
        Recording.usages.push(processes);
      };
    }

    if (Recording.count === 0){
      chrome.processes.onUpdatedWithMemory.addListener(Recording.listeningToProcesses);
    }
    ++Recording.count;
  }

  static _stopMonitoringProcess(tab){
    if (Recording.count === void 0 || Recording.count === 0)
      throw new Error("error");

    --Recording.count;
    if (Recording.count === 0){
      chrome.processes.onUpdatedWithMemory.removeListener(Recording.listeningToProcesses);
    }
    return Recording.usages;
  }

  start(){
    Recording._startMonitoringProcess();
  }

  stop(){
    Recording._stopMonitoringProcess();
    return Recording.usages.map((processes) => {
      return processes.filter((process) => {
        const task = process.tasks[0];
        if (task.tabId !== void 0 && task.tabId !== this._tab.id)
          return false;
        return true;
      });
    });
  }
}

async function sendResultToServer(result){
  const title = result.title;
  const usage = JSON.stringify(result.usage);
  const message = result.message;

  const {uploadServer} = await getSettings(["uploadServer"]);
  const url = `http://${uploadServer}/upload?id=${title}`;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
  return await new Promise((resolve, reject) => {
    xhr.onreadystatechange = () => {
      resolve();
    };
    xhr.send(`usage=${usage}&message=${message}`);
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
    const p1 = webpageEvent(tab, "start");
    const p2 = webpageEvent(tab, "stop");
    const {webpageServer} = await getSettings("webpageServer");
    const url = `http://${webpageServer}/getWebpage?id=${i}`;
    tab = await loadUrlInTab(url, tab);
    let params;
    params = await p1;
    const {title} = params;
    const record = new Recording(tab);
    record.start();

    params = {};
    try {
      params = await Promise.race([
        p2, waiting(timeout)
      ]);
    } catch (e){
    }

    const {message} = params;
    const usage = record.stop();
    await sendResultToServer({title, usage, message});
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
