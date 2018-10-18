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
app.post("/upload", (req, resp) => {
	const id = req.query.id;
	const usage = req.body.usage;
	const message = req.body.message;

	console.log(message);
	fs.writeFileSync(path.resolve(args.output, `${id}.json`), usage);
	fs.writeFileSync(path.resolve(args.output, `${id}.log`), message);
	resp.end();
});

app.listen(args.port, "localhost");
console.error("started");
