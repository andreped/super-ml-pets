const { app, BrowserWindow } = require("electron");
const path = require("path");
const memoryjs = require('memoryjs');
const processName = "Super Auto Pets.exe";


const createWindow = () => {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
        },
    });

    win.loadFile("src/renderer/index.html");
};

app.whenReady().then(() => {
    createWindow();
});
