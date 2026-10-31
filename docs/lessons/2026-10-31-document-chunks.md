# Interview log append uses the shared exact-byte chunk protocol

Append only the new bytes through the invoker/RLS function inside the original lease/Stop transaction. Explicit reads verify base and ordered epoch chunks, rejecting missing or changed bytes. Base overwrite/clear resets the epoch. Requires delivery-api's owner-reviewed document-chunk schema; do not deploy this runtime before it. Real PG coverage is in delivery's generator-document and installed-child acceptance suites.
