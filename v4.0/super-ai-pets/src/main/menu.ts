import {
    Menu,
    shell,
    BrowserWindow,
  } from 'electron';
  
  export default class MenuBuilder {
    mainWindow: BrowserWindow;
  
    constructor(mainWindow: BrowserWindow) {
      this.mainWindow = mainWindow;
    }
  
    buildMenu(): Menu {
      if (
        process.env.NODE_ENV === 'development' ||
        process.env.DEBUG_PROD === 'true'
      ) {
        this.setupDevelopmentEnvironment();
      }
  
      const template = this.buildDefaultTemplate();
  
      const menu = Menu.buildFromTemplate(template);
      Menu.setApplicationMenu(menu);
  
      return menu;
    }
  
    setupDevelopmentEnvironment(): void {
      this.mainWindow.webContents.on('context-menu', (_, props) => {
      });
    }
  
    buildDefaultTemplate() {
      const templateDefault = [
        {
          label: '&SAP',
          submenu: [
            {
              label: '&Open',
              accelerator: 'Ctrl+O',
            },
            {
              label: '&Close',
              accelerator: 'Ctrl+W',
              click: () => {
                this.mainWindow.close();
              },
            },
          ],
        },
        {
          label: '&AI',
          submenu: [
                  {
                    label: '&Reload',
                    accelerator: 'Ctrl+R',
                    click: () => {
                      this.mainWindow.webContents.reload();
                    },
                  },
                ]
        },
        {
          label: 'Help',
          submenu: [
            {
              label: 'Learn More',
              click() {
                shell.openExternal('https://www.youtube.com/c/HJKCreations');
              },
            },
            {
              label: 'Documentation',
              click() {
                shell.openExternal(
                  'https://github.com/HJK-Z/Super-Auto-Pets'
                );
              },
            },
          ],
        },
      ];
  
      return templateDefault;
    }
  }