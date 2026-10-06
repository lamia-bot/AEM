MEA Analyzer v8 — Build 22 | Windows EXE instructions

This is the same Build 22 program you tested on macOS. The Windows build
uses PyInstaller in ONEDIR mode; keep the generated EXE and _internal folder together.

BUILD WITH GITHUB ACTIONS

1. Unzip the source archive, then open the MEA_Analyzer_v8_22 folder.
2. In Finder press Command+Shift+. to display the hidden .github folder.
3. In your existing GitHub repository use Add file > Upload files.
4. Upload the CONTENTS of MEA_Analyzer_v8_22 (not the outer folder).
   Important: .github/workflows/build-windows.yml must be at that exact path.
5. Commit the files. Existing same-name files should be updated/replaced.
6. Open GitHub repo > Actions > Build Windows MEA Analyzer > Run workflow.
7. Once the build succeeds, download the artifact named
   MEA_Analyzer_Windows_Build22.
8. Unzip the downloaded artifact. If it contains a nested ZIP named
   MEA_Analyzer_Windows_Build22.zip, unzip that too.
9. Keep MEA_Analyzer/ together and double-click MEA_Analyzer.exe inside.

Windows SmartScreen may warn because the EXE is unsigned.
This ZIP is SOURCE + Windows build configuration, not itself an .exe.

No scientific methods or GUI behaviors were modified for Windows packaging.
