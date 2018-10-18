"use strict";

const path = require("path");
const fs = require("fs");

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

let usages = [];
let message = void 0;
app.post("/uploadUsage", (req, resp) => {
	const id = req.query.id;
	let usage;
	try{
		usage = JSON.parse(req.body.usage);	
	} catch (e){
		console.error(usage);
		return ;
	}
	usages.push(usage);
	resp.end();
});

app.post("/uploadMessage", (req, resp) => {
	const id = req.query.id;
	message = req.body.message;
	console.error(message);
	fs.writeFileSync(path.resolve(args.output, `${id}.json`), JSON.stringify(usages));
	usages.length = 0;
	fs.writeFileSync(path.resolve(args.output, `${id}.log`), message);
	message = void 0;
	resp.end();
})

app.listen(args.port, "localhost");
console.error("started");
