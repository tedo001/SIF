; Inno Setup script for the SIF Insight Console (Windows installer).
;
; Built by CI as:
;   iscc /DAppVersion=%VERSION% packaging\installer.iss
;
; The version always comes from the git tag via /DAppVersion, so the installer's
; "Programs and Features" entry, the app's own version and the tag agree.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "SIF Insight Console"
#define AppPublisher "Oil India Limited"
#define AppExeName "SIFConsole.exe"

[Setup]
AppId={{4C7F2E1A-9B3D-4E6F-8A21-5C0D7E9F1B34}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\SIF Insight Console
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=SIF-Console-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-machine when elevated, per-user otherwise: a plant workstation is often locked down.
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName} {#AppVersion}
; Let an update overwrite the previous install without a manual uninstall.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\SIFConsole\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
