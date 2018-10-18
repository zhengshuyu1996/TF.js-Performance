document.addEventListener("started", (e) => {
	chrome.runtime.sendMessage({
		"method": "start",
		"params": {
			"title": document.title
		}
	});
});

document.addEventListener("finished", (e) => {
	let detail = e.detail;
	if (detail == null) detail = {};

	chrome.runtime.sendMessage({
		"method": "stop",
		"params": {
			"message": e.detail.message
		}
	});
});
