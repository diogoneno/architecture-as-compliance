"""aac.prisma (Phase 6, Appendix C PRISMA close-out): the screening workbook, its counts and Figure C.1.

Hands-off rule: the analyst's records under appendix-c/run-*/ and
appendix-c/_superseded/, chapters/ and tools/ are read-only inputs. This package writes only to two roots:

 appendix-c/screening/ the workbook of record, batch files, pilot and gate records, the run-folder
 manifest (plus the one appendix-c/reference-verification-YYYY-MM-DD.csv);
 aac-pipeline/reports/appendix-c/ generated tables and figures for the paste-ready text.

Every writer goes through aac.prisma._io, which refuses any other target before a byte is written.
The pipeline prepares sheets and counts; the analyst decides every row (no decision cell is ever filled here).
"""
