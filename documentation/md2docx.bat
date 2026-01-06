@echo off
setlocal enabledelayedexpansion

:: Create temporary file for title
echo --- > title.md
echo title: "KredoBank Call Transcription Results" >> title.md
echo author: "Technical Team" >> title.md
echo date: "%date%" >> title.md
echo --- >> title.md
echo. >> title.md
echo. >> title.md

:: Start merged file with the title
copy title.md merged_documentation.md

:: Append numeric markdown files (0.md to 20.md) with page breaks
echo.
echo Adding numbered files to documentation...
for /L %%i in (0,1,20) do (
    if exist %%i.md (
        echo. >> merged_documentation.md
        echo. >> merged_documentation.md
        echo ^<div style="page-break-before: always;"^>^</div^> >> merged_documentation.md
        echo. >> merged_documentation.md
        type %%i.md >> merged_documentation.md
        echo Added: %%i.md with page break
    ) else (
        echo Skipping missing file: %%i.md
    )
)

:: Add CRM Summary Documentation at the beginning
if exist crm_summary_documentation_en.md (
    echo. >> merged_documentation.md
    echo. >> merged_documentation.md
    echo ^<div style="page-break-before: always;"^>^</div^> >> merged_documentation.md
    echo. >> merged_documentation.md
    type crm_summary_documentation_en.md >> merged_documentation.md
    echo Added: crm_summary_documentation_en.md
) else (
    echo Warning: crm_summary_documentation_en.md not found!
)

:: Add Call Evaluation Documentation
if exist call_evaluation_documentation_en.md (
    echo. >> merged_documentation.md
    echo. >> merged_documentation.md
    echo ^<div style="page-break-before: always;"^>^</div^> >> merged_documentation.md
    echo. >> merged_documentation.md
    type call_evaluation_documentation_en.md >> merged_documentation.md
    echo Added: call_evaluation_documentation_en.md
) else (
    echo Warning: call_evaluation_documentation_en.md not found!
)



:: Generate DOCX using pandoc with proper markdown and HTML processing
echo.
echo Generating DOCX file...
pandoc merged_documentation.md -f markdown+raw_html+yaml_metadata_block -t docx --toc --toc-depth=2 -o kredobank_call_evaluation.docx --reference-doc=reference.docx

:: Alternative method if the above doesn't work - try with different format options
:: pandoc merged_documentation.md -f markdown+raw_html+pandoc_title_block -t docx --toc --toc-depth=2 -o kredobank_call_evaluation.docx --reference-doc=reference.docx

:: Clean up temporary files
del title.md
del merged_documentation.md

echo.
echo Documentation has been generated:
echo - DOCX: kredobank_call_evaluation.docx
echo.
echo Files included in document:
echo 1. Title page
echo 2. CRM Summary Documentation (crm_summary_documentation_en.md)
echo 3. Call Evaluation Documentation (call_evaluation_documentation_en.md)
echo 4. Call Evaluation FN Documentation (call_evaluation_fn_documentation_en.md)
echo 5. Numbered files (0.md to 20.md - if present) - each with page break

echo.
echo Note: If markdown formatting is still not working, try one of these alternatives:
echo 1. Check that your .md files contain proper markdown syntax
echo 2. Ensure pandoc is properly installed and updated
echo 3. Try the alternative pandoc command in the script

pause