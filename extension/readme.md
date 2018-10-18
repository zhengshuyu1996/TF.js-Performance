requirement:

1. npm local install yargs and express `npm install <name>`
2. load unpacked extension to chrome from `./processes`

steps:

1. enter directory which contains html files and start puer `puer --allow-cors` (default )
2. add all html files’ URL to  a json file `./webpages/webpages.json`
3. enter `extension/webpages` and type command `node getWebpageServer.js -p <port>`
4. enter `extension` and type command `node uploadServer.js --output <output dir> --port <port>` 
5. start chrome, refresh extension and modify config (right click extension button and choose `option`)

