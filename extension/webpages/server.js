"use strict";

const path = require("path");
const fs = require("fs");
const child_process = require("child_process");

const yargs = require("yargs");
const express = require("express");
const body_parser = require("body-parser");

const args = yargs
	.string("output").alias("o", "output")
	.demandOption("output")
	.number("port").alias("p", "port")
	.default("port", 8001)
	.help()
	.argv;

const webpageList = require("./webpages.json");

const app = express();
app.get("/getWebpage", (req, resp) => {
	const id = parseInt(req.query.id);
	//resp.end(fs.readFileSync(path.resolve(__dirname, "index.html")));
	//return ;
	if (id >= 0 && id < webpageList.length)
		resp.redirect(webpageList[id]);
	else
		resp.status(404);
		//.send("<!DOCTYPE html><html><head><title>Error!</title></head><body>Webpage Not Found!</body></html>");
});

app.use(body_parser.urlencoded({ extended: false }));
app.use(body_parser.json());

function getGPUProcess(){
	let result = child_process.spawnSync("nvidia-smi");
	let lines = result.stdout.toString('utf8').split('\n');
	let gpu_process = [];
	let cnt = 0;
	for (let line of lines){
		if (cnt == 2){
			let frags = line.split(" ").filter(frag => frag !== "");
			if (frags.length == 7){
				gpu_process.push({
					pid: parseInt(frags[2]),
					memory: line[5]
				});
			}
		} else if (line[0] === '|' && line[1] === '=')
			cnt += 1;
	}
	return gpu_process;
}

let usages = [];
let message = void 0;
app.post("/uploadUsage", (req, resp) => {
	//console.log("hahahaha");
	const id = req.query.id;
	const usage = JSON.parse(req.body.usage);
	const tmp = {
		"chrome": usage
	};

	const proc = usage.find((proc) => proc.type === "gpu");
	if (proc !== void 0){
		const procId = proc.osProcessId;
		//console.log(procId);

		const gpu_process = getGPUProcess();
		const gproc = gpu_process.find((proc) => proc.pid === procId);
	
		tmp.gpu = gproc;
	}
	usages.push(tmp);
	resp.end();
});

app.post("/uploadMessage", (req, resp) => {
	//const id = req.query.id;
	message = req.body.message;
	console.log(message);
	
	let id = message.split('\t').join(" ");
	fs.writeFileSync(path.resolve(args.output, `${id}.json`), JSON.stringify(usages));
	usages.length = 0;
	//fs.writeFileSync(path.resolve(args.output, `${id}.log`), message);
	//message = void 0;
	//console.error("finish");
	resp.end();
})

app.listen(args.port, "localhost");
//console.error("started");
