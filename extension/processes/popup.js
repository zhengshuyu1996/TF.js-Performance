"use strict";

const create = document.getElementById("create");
create.addEventListener("click", () => {
	chrome.runtime.sendMessage({
		"method": "trigger",
		"params": {}
	});
});
