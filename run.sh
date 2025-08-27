#!/usr/bin/env bash

# -------------------------------
# usage:   ./run.sh -d /path/to/data_root
# example: ./run.sh -d /home/esahin/Data
# This runs CANTS Case-3 (pickle) on: $data_dir/hybrid_pytorch.pkl
# -------------------------------

# ------------ CLI: -d for data root ------------
while getopts "d:" opt; do
  case $opt in
    d) data_dir="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2 ;;
  esac
done
: "${data_dir:=/home/esahin/Data}"   # default if not provided

# =========================================================
# ===============  CANTS Case-3 (PICKLE)  =================
# =========================================================
export CANTS_CASE3=1
export CANTS_CASE3_ACC_THRESH=90.0
# (colonies.py also honors FITNESS_GLOBAL if that name preferred)
export FITNESS_GLOBAL="$CANTS_CASE3_ACC_THRESH"

# Optional: restrict the date window used by Case-3
export CANTS_CASE3_START=2012-01-01
export CANTS_CASE3_END=2019-12-31

# Optional: choose loss & promotion policy for candidates
#   CANTS_CASE3_LOSS ∈ {bicross_entropy|cross_entropy|mse}
#   CANTS_CASE3_PROMOTE_DELTA = required improvement (percentage points) over current acc (or threshold)
export CANTS_CASE3_LOSS=mse
export CANTS_CASE3_PROMOTE_DELTA=0.5

# Pickle file name (under $data_dir). Adjust if needed.
PICKLE_FILE="hybrid_pytorch.pkl"

# ---------------- Input features ----------------
#  - 'HighestModDate' or 'Malware' not included here.
#  - 'Malware' is the sole output and is passed via --output_names.
#  - Keep each feature on its own line; we join them into a space-separated string.

readarray -t FEATS <<'EOF'
execve
getuid32
getgid32
geteuid32
getegid32
getresuid32
getresgid32
readahead
getgroups32
getpgid
getppid
getsid
setsid
setgid32
setuid32
setreuid32
setresuid32
setresgid32
brk
kill
tgkill
ptrace
getrusage
getpriority
setpriority
ugetrlimit
setrlimit
prlimit64
setgroups32
setpgid
setregid32
chroot
prctl
capget
capset
sigaltstack
acct
read
write
pread64
pwrite64
preadv
pwritev
close
getpid
munmap
mremap
msync
mprotect
madvise
mlock
munlock
mlockall
munlockall
mincore
ioctl
readv
writev
fcntl64
flock
fchmod
dup
pipe2
dup3
fsync
fdatasync
fchown32
sync
fsetxattr
fgetxattr
flistxattr
fremovexattr
getdents64
openat
faccessat
fchmodat
fchownat
fstatat64
linkat
mkdirat
mknodat
readlinkat
renameat
symlinkat
unlinkat
utimensat
lseek
_llseek
ftruncate64
sendfile
sendfile64
truncate
truncate64
mmap2
fallocate
fadvise64_64
fstatfs64
statfs64
fstat64
chdir
mount
umount2
getcwd
fchdir
setxattr
lsetxattr
getxattr
lgetxattr
listxattr
llistxattr
removexattr
lremovexattr
swapon
swapoff
settimeofday
times
nanosleep
clock_settime
clock_getres
clock_nanosleep
getitimer
setitimer
timer_create
timer_settime
timer_gettime
timer_getoverrun
timer_delete
timerfd_create
timerfd_settime
timerfd_gettime
adjtimex
clock_adjtime
sigaction
rt_sigaction
rt_sigpending
rt_sigprocmask
rt_sigsuspend
rt_sigtimedwait
rt_sigqueueinfo
signalfd4
socket
bind
connect
listen
getsockname
getpeername
socketpair
sendto
recvfrom
shutdown
setsockopt
getsockopt
sendmsg
recvmsg
accept4
recvmmsg
sendmmsg
sched_setscheduler
sched_getscheduler
sched_yield
sched_setparam
sched_getparam
sched_get_priority_max
sched_get_priority_min
sched_rr_get_interval
sched_setaffinity
setns
unshare
sched_getaffinity
getcpu
uname
umask
reboot
init_module
delete_module
syslog
sysinfo
personality
tee
splice
vmsplice
epoll_create1
epoll_ctl
epoll_pwait
eventfd2
exit_group
exit
inotify_init1
inotify_add_watch
inotify_rm_watch
pselect6
ppoll
process_vm_readv
set_tid_address
setfsgid
setfsuid
sethostname
wait4
waitid
set_thread_area
clock_gettime
gettimeofday
clone
futex
vfork
rt_sigreturn
restart_syscall
getrandom
gettid
epoll_wait
stat64
pipe
pread
getrlimit
pwrite
recv
open
syscall_983042
SYS_300
SYS_301
SYS_302
SYS_303
SYS_304
SYS_305
SYS_306
SYS_307
SYS_308
SYS_309
SYS_310
SYS_311
SYS_312
SYS_313
SYS_314
SYS_315
SYS_316
SYS_317
SYS_318
SYS_319
SYS_320
SYS_321
SYS_322
SYS_323
SYS_324
SYS_325
SYS_326
SYS_327
SYS_328
SYS_329
SYS_330
SYS_331
SYS_332
SYS_333
SYS_334
SYS_335
SYS_336
SYS_337
SYS_338
SYS_339
SYS_340
SYS_341
SYS_342
SYS_343
SYS_344
SYS_345
SYS_346
SYS_347
SYS_348
SYS_349
SYS_350
SYS_351
SYS_352
SYS_353
SYS_354
SYS_355
SYS_356
SYS_357
SYS_358
SYS_359
SYS_360
SYS_361
SYS_362
SYS_363
SYS_364
SYS_365
SYS_366
SYS_367
SYS_368
SYS_369
ACCEPT_HANDOVER
ACCESS_BACKGROUND_LOCATION
ACCESS_CHECKIN_PROPERTIES
ACCESS_COARSE_LOCATION
ACCESS_FINE_LOCATION
ACCESS_LOCATION_EXTRA_COMMANDS
ACCESS_MEDIA_LOCATION
ACCESS_NETWORK_STATE
ACCESS_NOTIFICATION_POLICY
ACCESS_WIFI_STATE
ACCOUNT_MANAGER
ACTIVITY_RECOGNITION
ADD_VOICEMAIL
ANSWER_PHONE_CALLS
BATTERY_STATS
BIND_ACCESSIBILITY_SERVICE
BIND_APPWIDGET
BIND_AUTOFILL_SERVICE
BIND_CALL_REDIRECTION_SERVICE
BIND_CARRIER_MESSAGING_CLIENT_SERVICE
BIND_CARRIER_MESSAGING_SERVICE
BIND_CARRIER_SERVICES
BIND_CHOOSER_TARGET_SERVICE
BIND_CONDITION_PROVIDER_SERVICE
BIND_CONTROLS
BIND_DEVICE_ADMIN
BIND_DREAM_SERVICE
BIND_INCALL_SERVICE
BIND_INPUT_METHOD
BIND_MIDI_DEVICE_SERVICE
BIND_NFC_SERVICE
BIND_NOTIFICATION_LISTENER_SERVICE
BIND_PRINT_SERVICE
BIND_QUICK_ACCESS_WALLET_SERVICE
BIND_QUICK_SETTINGS_TILE
BIND_REMOTEVIEWS
BIND_SCREENING_SERVICE
BIND_TELECOM_CONNECTION_SERVICE
BIND_TEXT_SERVICE
BIND_TV_INPUT
BIND_VISUAL_VOICEMAIL_SERVICE
BIND_VOICE_INTERACTION
BIND_VPN_SERVICE
BIND_VR_LISTENER_SERVICE
BIND_WALLPAPER
BLUETOOTH
BLUETOOTH_ADMIN
BLUETOOTH_PRIVILEGED
BODY_SENSORS
BROADCAST_PACKAGE_REMOVED
BROADCAST_SMS
BROADCAST_STICKY
BROADCAST_WAP_PUSH
CALL_COMPANION_APP
CALL_PHONE
CALL_PRIVILEGED
CAMERA
CAPTURE_AUDIO_OUTPUT
CHANGE_COMPONENT_ENABLED_STATE
CHANGE_CONFIGURATION
CHANGE_NETWORK_STATE
CHANGE_WIFI_MULTICAST_STATE
CHANGE_WIFI_STATE
CLEAR_APP_CACHE
CONTROL_LOCATION_UPDATES
DELETE_CACHE_FILES
DELETE_PACKAGES
DIAGNOSTIC
DISABLE_KEYGUARD
DUMP
EXPAND_STATUS_BAR
FACTORY_TEST
FOREGROUND_SERVICE
GET_ACCOUNTS
GET_ACCOUNTS_PRIVILEGED
GET_PACKAGE_SIZE
GET_TASKS
GLOBAL_SEARCH
INSTALL_LOCATION_PROVIDER
INSTALL_PACKAGES
INSTALL_SHORTCUT
INSTANT_APP_FOREGROUND_SERVICE
INTERACT_ACROSS_PROFILES
INTERNET
KILL_BACKGROUND_PROCESSES
LOADER_USAGE_STATS
LOCATION_HARDWARE
MANAGE_DOCUMENTS
MANAGE_EXTERNAL_STORAGE
MANAGE_OWN_CALLS
MASTER_CLEAR
MEDIA_CONTENT_CONTROL
MODIFY_AUDIO_SETTINGS
MODIFY_PHONE_STATE
MOUNT_FORMAT_FILESYSTEMS
MOUNT_UNMOUNT_FILESYSTEMS
NFC
NFC_PREFERRED_PAYMENT_INFO
NFC_TRANSACTION_EVENT
PACKAGE_USAGE_STATS
PERSISTENT_ACTIVITY
PROCESS_OUTGOING_CALLS
QUERY_ALL_PACKAGES
READ_CALENDAR
READ_CALL_LOG
READ_CONTACTS
READ_EXTERNAL_STORAGE
READ_INPUT_STATE
READ_LOGS
READ_PHONE_NUMBERS
READ_PHONE_STATE
READ_PRECISE_PHONE_STATE
READ_SMS
READ_SYNC_SETTINGS
READ_SYNC_STATS
READ_VOICEMAIL
REBOOT
RECEIVE_BOOT_COMPLETED
RECEIVE_MMS
RECEIVE_SMS
RECEIVE_WAP_PUSH
RECORD_AUDIO
REORDER_TASKS
REQUEST_COMPANION_RUN_IN_BACKGROUND
REQUEST_COMPANION_USE_DATA_IN_BACKGROUND
REQUEST_DELETE_PACKAGES
REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
REQUEST_INSTALL_PACKAGES
REQUEST_PASSWORD_COMPLEXITY
RESTART_PACKAGES
SEND_RESPOND_VIA_MESSAGE
SEND_SMS
SET_ALARM
SET_ALWAYS_FINISH
SET_ANIMATION_SCALE
SET_DEBUG_APP
SET_PREFERRED_APPLICATIONS
SET_PROCESS_LIMIT
SET_TIME
SET_TIME_ZONE
SET_WALLPAPER
SET_WALLPAPER_HINTS
SIGNAL_PERSISTENT_PROCESSES
SMS_FINANCIAL_TRANSACTIONS
START_VIEW_PERMISSION_USAGE
STATUS_BAR
SYSTEM_ALERT_WINDOW
TRANSMIT_IR
UNINSTALL_SHORTCUT
UPDATE_DEVICE_STATS
USE_BIOMETRIC
USE_FINGERPRINT
USE_FULL_SCREEN_INTENT
USE_SIP
VIBRATE
WAKE_LOCK
WRITE_APN_SETTINGS
WRITE_CALENDAR
WRITE_CALL_LOG
WRITE_CONTACTS
WRITE_EXTERNAL_STORAGE
WRITE_GSERVICES
WRITE_SECURE_SETTINGS
WRITE_SETTINGS
WRITE_SYNC_SETTINGS
WRITE_VOICEMAIL
EOF

# Join features with spaces
IN_COLS="${FEATS[*]}"

# -------------------------------------------------
# Portable mpiexec invocation (Open MPI or MPICH)
# -------------------------------------------------
ENVLIST="CANTS_CASE3,CANTS_CASE3_ACC_THRESH,FITNESS_GLOBAL,CANTS_CASE3_START,CANTS_CASE3_END,CANTS_CASE3_LOSS,CANTS_CASE3_PROMOTE_DELTA"

MPI_IMPL="$(mpiexec --version 2>&1 | head -n1)"
MPI_MISC_ARGS=()
MPI_ENV_ARGS=()

if [[ "$MPI_IMPL" == *"Open MPI"* ]]; then
  # Open MPI: use --oversubscribe and -x VAR
  MPI_MISC_ARGS+=(--oversubscribe)
  IFS=',' read -r -a _vars <<< "$ENVLIST"
  for v in "${_vars[@]}"; do MPI_ENV_ARGS+=(-x "$v"); done
else
  # MPICH / Intel MPI (Hydra): use -genvlist
  MPI_ENV_ARGS=(-genvlist "$ENVLIST")
fi

echo "==> Running CANTS Case-3 (pickle, MPI x5) on: ${data_dir}/${PICKLE_FILE}"
mpiexec -n 5 "${MPI_MISC_ARGS[@]}" "${MPI_ENV_ARGS[@]}" \
  python3 ./src/colonies.py \
    --data_dir "${data_dir}" \
    --data_files "${PICKLE_FILE}" \
    --input_names "${IN_COLS}" \
    --output_names "Malware" \
    --out_dir OUT --log_dir LOG \
    --living_time 90 --comm_interval 2 \
    --term_log_level INFO --log_file_name cants_case3_pickle \
    --file_log_level INFO --col_log_level INFO \
    --use_bp --bp_epochs 9 --loss_fun mse -nrm minmax

exit 0
