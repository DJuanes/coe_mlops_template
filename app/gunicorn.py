import os

#
# Server socket
#
#   bind - El socket a enlazar.
#
#       Una cadena de la forma: 'HOST', 'HOST:PORT', 'unix:PATH'.
#       Una IP es un HOST valido.
#
#   backlog - el número de conexiones pendientes. Esto se refiere
#       a la cantidad de clientes que pueden estar esperando para ser
#       servido. Si se supera este número, el cliente recibe un error
#       al intentar conectarse. Solo debería afectar a los servidores
#       bajo una carga significativa.
#
#       Debe ser un entero positivo.
#       Generalmente establecido en el rango 64-2048
#

port = os.environ.get("PORT", 8000)
bind = f"0.0.0.0:{port}"
# backlog = 2048

#
# Worker processes
#
#   workers - La cantidad de procesos de trabajo que este servidor
#       debe mantener activos para manejar las solicitudes.
#
#       Un entero positivo generalmente en el rango de 2-4 x $(NUM_CORES).
#       Querrá variar esto un poco para encontrar lo mejor para
#       la carga de trabajo de su aplicación en particular.
#
#   worker_class - El tipo de trabajadores que se usarán.
#       La clase de sincronización predeterminada debe manejar
#       la mayoría de los tipos de cargas de trabajo "normales".
#       Querrás leer
#       http://docs.gunicorn.org/en/latest/design.html#choosing-a-worker-type
#       para obtener información sobre cuándo es posible que desee elegir
#       una de las otras clases de trabajadores.
#
#       Una cadena que hace referencia a una ruta de Python a una subclase
#       de gunicorn.workers.base.Worker. Los valores proporcionados por defecto
#       se pueden ver en
#       http://docs.gunicorn.org/en/latest/settings.html#worker-class
#
#   worker_connections - Para las clases de trabajadores eventlet y gevent,
#       esto limita la cantidad máxima de clientes simultáneos
#       que puede manejar un solo proceso.
#
#       Un entero positivo generalmente establecido en alrededor de 1000.
#
#   timeout - Si un trabajador no notifica al proceso maestro
#       en esta cantidad de segundos, se elimina y se genera
#       un nuevo trabajador para reemplazarlo.
#
#       Generalmente se establece en treinta segundos.
#       Establezca esto notablemente más alto solo si está seguro
#       de las repercusiones para los trabajadores de sincronización.
#       Para los trabajadores que no están sincronizados,
#       solo significa que el proceso de trabajo aún se está comunicando
#       y no está vinculado al tiempo requerido para manejar una sola solicitud.
#
#   keepalive - La cantidad de segundos de espera para la siguiente solicitud
#       en una conexión HTTP Keep-Alive.
#
#       Un entero positivo. Generalmente se establece en el rango de 1 a 5 segundos.
#

# workers = multiprocessing.cpu_count() * 2 + 1
workers = 1
# worker_connections = 1000
timeout = 90
# keepalive = 2

#
#   spew - Instala una función de seguimiento que arroja cada línea de Python
#       que se ejecuta al ejecutar el servidor. Esta es la opción nuclear.
#
#       True o False
#
#   reload - Reinicia los trabajadores cuando cambia el código.
#       Esta configuración está pensada para el desarrollo.
#       Hará que los trabajadores se reinicien cada vez que cambie
#       el código de la aplicación.

spew = False
reload = False

#
# Server mechanics
#
#   daemon - Separa el proceso principal de Gunicorn del terminal de control
#       con una secuencia bifurcación/bifurcación estándar.
#
#       True o False
#
#   raw_env - Pasa variables de entorno al entorno de ejecución.
#
#   pidfile - La ruta a un archivo pid para escribir
#
#       Una cadena de ruta o None para no escribir un archivo pid.
#
#   user - Cambie los procesos de trabajo para que se ejecuten como este usuario.
#
#       Una identificación de usuario válida (como un número entero)
#       o el nombre de un usuario que se puede recuperar con una llamada
#       a pwd.getpwnam(value) o None para no cambiar el usuario del proceso de trabajo.
#
#   group - Cambie el proceso de trabajo para que se ejecute como este grupo.
#
#       Una identificación de grupo válida (como un número entero)
#       o el nombre de un usuario que se puede recuperar con una llamada
#       a pwd.getgrnam(value) o None para cambiar el grupo de procesos de trabajo.
#
#   umask - Una máscara para permisos de archivos escrita por Gunicorn.
#       Tenga en cuenta que esto afecta los permisos de socket de Unix.
#
#       Un valor válido para la llamada os.umask(mode) o una cadena
#       compatible con int(value, 0) (0 significa que Python adivina la base,
#       por lo que valores como "0", "0xFF", "0022" son válidos
#       para decimal, hexadecimal y octal)
#
#   tmp_upload_dir - Un directorio para almacenar datos de solicitudes temporales
#       cuando se leen solicitudes. Lo más probable es que esto desaparezca pronto.
#
#       Una ruta a un directorio donde el propietario del proceso puede escribir.
#       O None para indicar que Python debe elegir uno por su cuenta.
#

# daemon = False
raw_env = []
# pidfile = None
# umask = 0
# user = None
# group = None
# tmp_upload_dir = None

#
#   Logging
#
#   logfile - La ruta a un archivo log para escribir.
#
#       Una cadena de ruta. "-" significa iniciar sesión en stdout.
#
#   loglevel - La granularidad de la salida del log.
#
#       Una cadena de "debug", "info", "warning", "error", "critical"
#

errorlog = "-"
# loglevel = 'info'
accesslog = "-"
# access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

#
# Process naming
#
#   proc_name - Una base para usar con setproctitle para cambiar la forma
#       en que los procesos de Gunicorn se informan en la tabla de procesos del sistema.
#       Esto afecta cosas como 'ps' y 'top'. Si va a ejecutar más de una instancia
#       de Gunicorn, probablemente querrá establecer un nombre para diferenciarlas.
#       Esto requiere que instale el módulo setproctitle.
#
#       Una cadena o None para elegir un valor predeterminado de algo como 'gunicorn'.
#

proc_name = None

#
# Server hooks
#
#   post_fork - Llamado justo después de que un trabajador haya sido bifurcado.
#
#       Un invocable que toma una instancia de servidor y trabajador como argumentos.
#
#   pre_fork - Llamado justo antes de bifurcar el subproceso del trabajador.
#
#       Un invocable que acepta los mismos argumentos que after_fork
#
#   pre_exec - Llamado justo antes de bifurcar un proceso maestro secundario
#       durante cosas como la recarga de configuración.
#
#       Un invocable que toma una instancia de servidor como único argumento.
#


def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def pre_fork(server, worker):
    pass


def pre_exec(server):
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    server.log.info("Server is ready. Spawning workers")


def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

    # get traceback info
    import sys
    import threading
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    worker.log.debug("\n".join(code))


def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")
