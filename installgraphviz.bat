@REM this command will correctly install pygraphviz on windows (need to have graphviz installed!)
set "GRAPHVIZ_PATH=C:\Program Files\Graphviz"
py -m pip install --config-settings="--global-option=build_ext" ^
                    --config-settings="--global-option=-I%GRAPHVIZ_PATH%\include" ^
                    --config-settings="--global-option=-L%GRAPHVIZ_PATH%\lib" ^
                pygraphviz