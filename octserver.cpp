#include <iostream>
#include <QtWidgets>
#include <QtNetwork>
#include <QDir>
#include "config.h"

#include "octserver.h"

// TODO: Refactor to remove the fft reconstruction from the server.

OCTServer::OCTServer(QWidget *parent, int nx, int n_extra, unsigned int n_alines)
    : QDialog(parent)
    , statusLabel(new QLabel)
    , tcpServer(new QTcpServer),
      f_fft(1, 1) // TODO: use n_repeat and factor for the f_fft constructor
{
    statusLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
    p_request_type = REQUEST_TILE;
    p_nx = nx;
    p_n_extra = n_extra;
    p_n_alines = n_alines;
    p_put_done = false;


    initServer();
    auto quitButton = new QPushButton(tr("Quit"));
    quitButton->setAutoDefault(false);

    // Prepare the data reconstruction
    float dimz = 3.5; // FIXME: dummy axial resolution
    float dimx = 3.0; // FIXME: dummy lateral resolution
    f_fft.init(LINE_ARRAY_SIZE, p_nx + p_n_extra, dimz, dimx);
    p_fringe_buffer = new unsigned short[(p_nx + p_n_extra)*LINE_ARRAY_SIZE];
    p_image_buffer = new float[(p_nx + p_n_extra)*LINE_ARRAY_SIZE/2];

    // Connections
    connect(quitButton, &QAbstractButton::clicked, this, &QWidget::close);
    connect(tcpServer, &QTcpServer::newConnection, this, &OCTServer::slot_startConnection);

    // Layout
    auto buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(quitButton);
    buttonLayout->addStretch(1);

    QVBoxLayout *mainLayout = nullptr;
    if (QGuiApplication::styleHints()->showIsFullScreen() || QGuiApplication::styleHints()->showIsMaximized()) {
        auto outerVerticalLayout = new QVBoxLayout(this);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
        auto outerHorizontalLayout = new QHBoxLayout;
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        auto groupBox = new QGroupBox(QGuiApplication::applicationDisplayName());
        mainLayout = new QVBoxLayout(groupBox);
        outerHorizontalLayout->addWidget(groupBox);
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        outerVerticalLayout->addLayout(outerHorizontalLayout);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
    } else {
        mainLayout = new QVBoxLayout(this);
    }

    mainLayout->addWidget(statusLabel);
    mainLayout->addLayout(buttonLayout);

    setWindowTitle(QGuiApplication::applicationDisplayName());
}

OCTServer::~OCTServer()
{
    tcpServer->close();
}

void OCTServer::initServer()
{
    // Set the server address and port
    QString ipAddress = OCTServer::getHostAddress();
    const QHostAddress address = QHostAddress(ipAddress);
    quint16 ipPort = SERVER_SOCKET_PORT;

    if (!tcpServer->listen(address, ipPort)) {
        QMessageBox::critical(this, tr("OCT Server"),
                              tr("Unable to start the server: %1.")
                              .arg(tcpServer->errorString()));
        close();
        return;
    }

    statusLabel->setText(tr("The server is running on\n\nIP: %1\nport: %2\n\n"
                            "Waiting for connections ...")
                         .arg(ipAddress).arg(tcpServer->serverPort()));

}

void OCTServer::put(unsigned short* fringe)
{
    // TODO: Remove those hardcoded values
    int top_z = 1;
    int bottom_z = LINE_ARRAY_SIZE/2;
    int hanning_threshold = 100;

    if (p_mutex.tryLock())
    {
        if (!p_put_done) {
            std::cerr << "Updating put" << std::endl;
            memcpy(p_fringe_buffer, fringe, p_n_alines*LINE_ARRAY_SIZE*sizeof(unsigned short));
            f_fft.image_reconstruction(p_fringe_buffer, p_image_buffer, top_z, bottom_z, hanning_threshold );
            p_put_done = true;
        }
    }
    p_mutex.unlock();
}

void OCTServer::slot_endConnection()
{
    std::cerr << "octserver::Closing the connection...";
    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    out.setVersion(QDataStream::Qt_6_4);
    out.setFloatingPointPrecision(QDataStream::SinglePrecision);
    switch (p_request_type) {
    case REQUEST_TILE:
        out << "OCT_done";
        break;
    case REQUEST_BSCAN_NOSAVE:
        // send back the number of alines, z values, and then the data.
        int n = (p_nx + p_n_extra) * (LINE_ARRAY_SIZE/2);
        int n_bytes = n * sizeof(float);
        std::cerr << n << "," << n_bytes << std::endl;
        out << n_bytes;
        //out << p_image_buffer;
        for (int i=0; i<n; i++) {
           out << p_image_buffer[i];
        }
        break;
    }
    clientConnection->write(response);
    std::cerr << " done!" << std::endl;
    p_mutex.lock();
    p_put_done = false;
    p_mutex.unlock();
}

QString OCTServer::getHostAddress()
{
    QString ipAddress;
    const QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();

    // use the first non-localhost IPv4 address
    for (const QHostAddress &entry : ipAddressesList) {
        if (entry != QHostAddress::LocalHost && entry.toIPv4Address()) {
            ipAddress = entry.toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    return ipAddress;
}

void OCTServer::slot_parseRequest()
{
    QByteArray total_data, buffer;
    while(1) {
        buffer = clientConnection->read(1024);
        if (buffer.isEmpty()) {
            break;
        }
        total_data.append(buffer);
    }
    QString data = QString(total_data);

   if ( data.startsWith("request_autofocus_bscan"))
   {
       std::cerr << "Case 1: Autofocus B-Scan Request" << std::endl;
       p_request_type = REQUEST_BSCAN_NOSAVE;
       emit sig_set_request_type(QString("autofocus"));
       setup_and_request_scan(-1, -1, -1);
   }
   else
   {
       std::cerr << "Case 2: Tile Request" << std::endl;
       p_request_type = REQUEST_TILE;
       emit sig_set_request_type(QString("tile"));
       std::cerr << "octserver::Reading tile positions: (x,y,z)=";
       // Read the tile position to acquire
       p_tile_x = data.split(" ")[0].toInt();
       p_tile_y = data.split(" ")[1].toInt();
       p_tile_z = data.split(" ")[2].toInt();

       std::cerr << "(" << p_tile_x <<"," << p_tile_y << "," << p_tile_z << ")" << std::endl;
        setup_and_request_scan(p_tile_x, p_tile_y, p_tile_z);
   }
}


void OCTServer::slot_startConnection()
{
    std::cerr << "octserver::Fetching the next pending connection...";
    clientConnection = tcpServer->nextPendingConnection();
    std::cerr << "got it!" << std::endl;

    connect(clientConnection, &QIODevice::readyRead, this, &OCTServer::slot_parseRequest);
}

void OCTServer::setup_and_request_scan(int x, int y, int z)
{
    // Create the tile filename
    char buffer [20];
    snprintf(buffer, 20, "tile_x%02d_y%02d_z%02d", x, y, z);
    QString fileName = QString(buffer);
    emit sig_change_filename(fileName);

    // Launch a single acquisition
    emit sig_start_scan();
}
